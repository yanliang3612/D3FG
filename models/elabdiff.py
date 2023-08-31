from .model_loader import register_model
import torch
import torch.nn as nn
from .diffusion.fgdpm import FGDPM
from .diffusion.linkerdpm import LinkerDPM
from .utils.geometry import *
from .utils.so3 import *
from .embeddings import *
from datasets.protein.constants import *
from datasets.protein.constants import max_num_heavyatoms, BBHeavyAtom
from datasets.molecule.constants import *
from datasets.protein.constants import num_aa_types
from torch.nn.utils.rnn import pad_sequence
from .fgdiff import FunctionalGroupDiff

num_total_type = num_fg_types + num_aa_types + num_atom_types

@register_model('elabdiff')
class MolElabDiff(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.target = self.cfg.target

        num_atoms = max_num_heavyatoms

        max_context_type = num_total_type

        self.num_steps = self.cfg.diffusion.num_steps

        self.residue_embed = ResidueEmbedding(
            cfg.fg_feat_dim, num_atoms, max_aa_types=max_context_type
            )
        self.pair_embed = PairEmbedding(
            cfg.pair_feat_dim, num_atoms, max_aa_types=max_context_type
            )

        if self.target == 'linker':
            linker_trans_type_opt = {
                'min_type_num':num_aa_types, 
                'max_type_num':num_aa_types + num_atom_types, 
                'num_classes':num_total_type
            }

            cfg.diffusion['trans_type_opt'] = linker_trans_type_opt
            self.linker_diffusion = FGDPM(
                cfg.fg_feat_dim,
                cfg.pair_feat_dim,
                **cfg.diffusion,
            )
        
        elif self.target == 'fg':
            fg_trans_type_opt = {
                'min_type_num':num_aa_types + num_atom_types, 
                'max_type_num':num_aa_types + num_atom_types + num_fg_types, 
                'num_classes':num_total_type
            }

            cfg.diffusion['trans_type_opt'] = fg_trans_type_opt
            self.fg_diffusion = FGDPM(
                cfg.fg_feat_dim,
                cfg.pair_feat_dim,
                **cfg.diffusion,
            )

    
    def sum_weighted_losses(self, losses, weights):
        """
        Args:
            losses:     Dict of scalar tensors.
            weights:    Dict of weights.
        """
        loss = 0
        for k in losses.keys():
            if weights is None:
                loss = loss + losses[k]
            else:
                loss = loss + weights[k] * losses[k]
        return loss

    def encode(self, batch, remove_structure, remove_type):
        """
        Returns:
            res_feat:   (N, L, fg_feat_dim)
            pair_feat:  (N, L, L, pair_feat_dim)
        """
        # This is used throughout embedding and encoding layers
        #   to avoid data leakage.
        context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA].bool(), 
            ~batch['fg_flag'].bool()     # Context means ``not generated''
        )

        structure_mask = context_mask if remove_structure else None
        type_mask = context_mask if remove_type else None

        res_feat = self.residue_embed(
            aa = batch['fg_type'],
            res_nb = batch['res_nb'],
            chain_nb = batch['seq_type'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            fragment_type = batch['seq_type'],
            structure_mask = structure_mask,
            type_mask = type_mask,
        )

        pair_feat = self.pair_embed(
            aa = batch['fg_type'],
            res_nb = batch['res_nb'],
            chain_nb = batch['seq_type'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            structure_mask = structure_mask,
            type_mask = type_mask,
        )

        R = construct_3d_basis_include_single_atom(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        return res_feat, pair_feat, R, p        

    def loss_single_step(self, batch, t=None):
        
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = self.cfg.get('train_structure', True),
            remove_type = self.cfg.get('train_type', True)
        )
        if self.target == 'fg':
            mask_fg = batch['fg_flag']
            mask_fg_context = batch['fg_context_flag']
            v_0 = rotation_to_so3vec(R_0)
            s_0_fg = batch['fg_type']

            loss_dict_fg = self.fg_diffusion(
                v_0, p_0, s_0_fg, res_feat, pair_feat, mask_fg, mask_fg_context,
                denoise_structure = self.cfg.get('train_structure', True),
                denoise_type = self.cfg.get('train_type', True),
                t = t
            )
            loss_dict = loss_dict_fg

        elif self.target == 'linker':
            mask_linker = batch['linker_flag']
            mask_linker_context =  batch['linker_context_flag']
            v_0 = rotation_to_so3vec(R_0)
            s_0_linker = batch['linker_type']

            loss_dict_linker = self.linker_diffusion(
                v_0, p_0, s_0_linker, res_feat, pair_feat, 
                mask_linker, mask_linker_context,
                denoise_structure = self.cfg.get('train_structure', True),
                denoise_type = self.cfg.get('train_type', True),
                t = t
            )

            loss_dict = loss_dict_linker
        
        loss = self.sum_weighted_losses(loss_dict, self.cfg.loss_weights)
        loss_dict['overall'] = loss 

        return loss, loss_dict
    
    def loss_all_step(self, batch, interval=None):
        if interval is None:
            interval = self.num_steps // 100
        eval_step_num = self.num_steps // interval
        ts = torch.linspace(0, self.num_steps, eval_step_num).long()
        losses, loss_dicts = [], []
        for t in ts:
            loss, loss_dict = self.loss_single_step(batch, t)
            losses.append(loss)
            loss_dicts.append(loss_dict)
        
        loss_all_step = torch.tensor(loss).mean()
        loss_dict_all_step = {key:[] for key,val in loss_dicts[0].items()}
        for loss_dict in loss_dicts:
            for key, val in loss_dict.items():
                loss_dict_all_step[key].append(val)

        for key, val in loss_dict_all_step.items():
            loss_dict_all_step[key] = torch.tensor(val).mean()
        
        return loss_all_step, loss_dict_all_step


    def forward(self, batch):
        if self.training:
            return self.loss_single_step(batch)
        else:
            with torch.no_grad():
                return self.loss_all_step(batch)

    
    @torch.no_grad()
    def translate_back(self, pos, com=None, mask=1):
        if com is not None:
            pos += com.unsqueeze(dim=1)
        pos = pos * mask.unsqueeze(dim=-1)
        return pos

    @torch.no_grad()
    def elaborate(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_type': True,
        }
    ):  
        
        if self.target == 'fg':
            res_feat, pair_feat, R_0, p_0 = self.encode(
                batch,
                remove_structure = self.cfg.get('train_structure', True),
                remove_type = self.cfg.get('train_type', True)
                )
            
            mask_fg = batch['fg_flag']
            mask_fg_context = batch['fg_context_flag']
            v_0 = rotation_to_so3vec(R_0)
            s_0_fg = batch['fg_type']

            (final_v_fg, final_pos_fg, final_s_fg), traj_fg = self.fg_diffusion.sample(
                    v_0, p_0, s_0_fg, res_feat, pair_feat, mask_fg, mask_fg_context, **sample_opt
                    )
            
            final_v, final_pos, final_s = (final_v_fg, final_pos_fg, final_s_fg)
            traj = {'fg':traj_fg}

            batch['gen_mask'] = mask_fg
            
        final_pos = self.translate_back(final_pos, batch['cond_com'], batch['gen_mask'])
        return batch, (final_v, final_pos, final_s), traj

    @torch.no_grad()
    def sample(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_type': True,
            'use_old_pos': False
        },
        pregen_model=None
    ):  
        
        if self.target == 'fg':
            sample_opt['use_old_pos'] = True

            res_feat, pair_feat, R_0, p_0 = self.encode(
                batch,
                remove_structure = self.cfg.get('train_structure', True),
                remove_type = self.cfg.get('train_type', True)
                )
            
            mask_fg = batch['fg_flag']
            mask_fg_context = batch['fg_context_flag']
            v_0 = rotation_to_so3vec(R_0)
            s_0_fg = batch['fg_type']

            (final_v_fg, final_pos_fg, final_s_fg), traj_fg = self.fg_diffusion.sample(
                    v_0, p_0, s_0_fg, res_feat, pair_feat, mask_fg, mask_fg_context, **sample_opt
                    )


            final_pos, final_s, atom_mask, is_aromatic = self.decode_batch_fg_to_atom(
                final_v_fg, final_pos_fg, final_s_fg, batch['ligand_flag']
                )
            batch['mol_mask'] = atom_mask
            batch['is_aromatic'] = is_aromatic
            (final_pos, final_s) = (final_pos, final_s + num_aa_types)
            traj = {'fg':traj_fg}

        else:
            res_feat, pair_feat, R_0, p_0 = self.encode(
                batch,
                remove_structure = self.cfg.get('train_structure', True),
                remove_type = self.cfg.get('train_type', True)
                )
            
            mask_linker = batch['linker_flag']
            mask_linker_context =  batch['linker_context_flag']
            v_0 = rotation_to_so3vec(R_0)
            s_0_linker = batch['linker_type']

            (final_v_linker, final_pos_linker, final_s_linker), traj_linker = self.linker_diffusion.sample(
                    v_0, p_0, s_0_linker, res_feat, pair_feat, mask_linker, mask_linker_context, **sample_opt
                    )

            batch['mol_mask'] = batch['ligand_flag']
            batch['is_aromatic'] = None

            (final_pos, final_s) = (final_pos_linker, final_s_linker)
            traj = {'linker': traj_linker}
        
        final_pos = self.translate_back(final_pos, batch['cond_com'], batch['mol_mask'])
        return batch, (final_pos, final_s), traj
    
    @torch.no_grad()
    def decode_batch_fg_to_atom(self, v, pos, fg, mask_fg):
        fg_atom_types, fg_atom_poses, fg_atom_aromatic, fg_atom_mask = [], [], [], []
        for current_v, current_pos, current_fg, current_mask in zip(v, pos, fg, mask_fg):
            atom_type, atom_pos, is_aromatic = self.decode_fg_to_atom(
                current_v[current_mask], 
                current_pos[current_mask], 
                current_fg[current_mask], 
                shift_back=False
                )
            fg_atom_types.append(atom_type)
            fg_atom_poses.append(atom_pos)
            fg_atom_aromatic.append(is_aromatic)
            atom_mask = torch.ones(len(atom_type)).to(atom_type).bool()
            fg_atom_mask.append(atom_mask)

            assert(len(atom_type) == len(atom_pos) == len(is_aromatic))

        batch_atom_pos = pad_sequence(fg_atom_poses, batch_first=True).float()
        batch_atom_type = pad_sequence(fg_atom_types, batch_first=True, padding_value=num_total_type).long()
        batch_atom_mask = pad_sequence(fg_atom_mask, batch_first=True).bool()
        batch_is_aromatic = pad_sequence(fg_atom_aromatic, batch_first=True).bool()
        return batch_atom_pos, batch_atom_type, batch_atom_mask, batch_is_aromatic


    @torch.no_grad()
    def decode_fg_to_atom(self, current_v, current_pos, current_fg, shift_back=True):
        device = current_v.device
        shift_class = num_aa_types
        current_fg = current_fg - shift_class
        fg_smiles = []
        for key in current_fg:
            if class2fg_dict[key.item()] == 'Others':
                fg_smiles.append('C') 
            else:
                fg_smiles.append(class2fg_dict[key.item()]) 
        
        fg_local_pos = [motif_pos_fractory[key] for key in fg_smiles]

        fg_smiles_raw = []
        for fg in fg_smiles:
            if fg in [ocno_chirality1, ocno_chirality2]:
                fg = 'O=CNO'
            elif fg in [nso2_chirality1, nso2_chirality2]:
                fg = 'NS(=O)=O'
            else:
                fg = fg
            fg_smiles_raw.append(fg)
        fg_atom_pos = []
        fg_atom_type = []
        fg_atom_aromatic = []

        for i in range(len(fg_smiles_raw)):
            R = so3vec_to_rotation(torch.tensor(current_v[i])).float().unsqueeze(dim=0)
            t = torch.tensor(current_pos[i]).float().unsqueeze(dim=0).to(current_v)
            p = torch.tensor(fg_local_pos[i]).float().unsqueeze(dim=0).to(current_v)
            global_fg_pos = local_to_global(R, t, p).tolist()[0]
            fg_atom_pos.append(global_fg_pos)

            fg_smiles = fg_smiles_raw[i]
            rd_fg = Chem.MolFromSmiles(fg_smiles)
            fg_atoms = []
            is_aromatic = []

            for atom in rd_fg.GetAtoms():
                element = atom.GetSymbol()
                if shift_back:
                    fg_atoms.append(fg2class_dict[element] + shift_class)
                else:
                    fg_atoms.append(fg2class_dict[element])
            fg_atom_type.append(fg_atoms)

            assert(len(global_fg_pos) == len(fg_atoms))

            for id, atom in enumerate(rd_fg.GetAtoms()):
                aromatic = rd_fg.GetAtomWithIdx(id).GetIsAromatic()
                is_aromatic.append(aromatic)
            fg_atom_aromatic.append(is_aromatic)

        if len(fg_atom_pos) > 0:
            fg_atom_pos = torch.from_numpy(np.concatenate(fg_atom_pos, axis=0))
            fg_atom_type = torch.from_numpy(np.concatenate(fg_atom_type, axis=0))
            fg_atom_aromatic = torch.from_numpy(np.concatenate(fg_atom_aromatic, axis=0))
        else:
            fg_atom_pos = torch.tensor(fg_atom_pos)
            fg_atom_type = torch.tensor(fg_atom_type)
            fg_atom_aromatic = torch.tensor(fg_atom_aromatic)
            
        return fg_atom_type.to(device), fg_atom_pos.to(device), fg_atom_aromatic.to(device)
    