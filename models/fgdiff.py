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

num_total_type = num_fg_types + num_aa_types + num_atom_types

@register_model('fgdiff')
class FunctionalGroupDiff(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.scheme = self.cfg.scheme
        self.num_steps = self.cfg.diffusion.num_steps

        num_atoms = max_num_heavyatoms

        max_context_type = 22

        self.residue_embed = ResidueEmbedding(
            cfg.fg_feat_dim, num_atoms, max_aa_types=max_context_type
            )
        self.pair_embed = PairEmbedding(
            cfg.pair_feat_dim, num_atoms, max_aa_types=max_context_type
            )

        if self.scheme == 'two_stage':
            self.linker_diffusion = LinkerDPM(
                cfg.node_feat_dim,
                cfg.pair_feat_dim * cfg.linker_protein_context,
                **cfg.diffusion,
            )
            fg_trans_type_opt = {
                'min_type_num':num_aa_types + num_atom_types, 
                'max_type_num':num_aa_types + num_atom_types + num_fg_types, 
                'num_classes':num_total_type
            }
            use_egnn_update = False
        
        elif self.scheme == 'joint':
            fg_trans_type_opt = {
                'min_type_num':num_aa_types, 
                'max_type_num':num_aa_types + num_atom_types + num_fg_types, 
                'num_classes':num_total_type
            }
            use_egnn_update = True


        cfg.diffusion['trans_type_opt'] = fg_trans_type_opt
        cfg.diffusion['eps_net_opt']['use_egnn_update'] = use_egnn_update
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
        
        mask_fg = batch['fg_flag']
        mask_fg_context = batch['fg_context_flag']

        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = self.cfg.get('train_structure', True),
            remove_type = self.cfg.get('train_type', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0_fg = batch['fg_type']

        loss_dict_fg = self.fg_diffusion(
            v_0, p_0, s_0_fg, res_feat, pair_feat, mask_fg, mask_fg_context,
            denoise_structure = self.cfg.get('train_structure', True),
            denoise_type = self.cfg.get('train_type', True),
            t = t
        )
        loss_dict = loss_dict_fg

        if self.scheme == 'two_stage':
            p_0_linker = batch['linker_pos']
            s_0_linker = batch['linker_type']
            mask_linker = batch['linker_flag']
            mask_linker_context =  batch['linker_context_flag']

            loss_dict_linker = self.linker_diffusion(
                p_0_linker, s_0_linker, mask_linker, mask_linker_context,
                denoise_structure = self.cfg.get('train_structure', True),
                denoise_type = self.cfg.get('train_type', True),
                t = t
            )

            loss_dict_fg.update(loss_dict_linker)
        
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
    def assign_com_and_fgsize_with_translate(self, batch, pregen_model_fg):
        batch_size, seq_len = batch['seq_type'].shape

        if pregen_model_fg is not None:
            fg_nums = pregen_model_fg(size=batch_size, batch=batch)
            coms = batch['cond_com']
        else:
            (coms, fg_nums ) = (
                batch['cond_com'], batch['fg_size']
            )

        batch['gen_com'], batch['fg_size'] = (
            coms, fg_nums,
        )

        batch['pos_heavyatom'] -= batch['gen_com'][:,None,None,:]
        batch['pos_heavyatom'] *= batch['mask_heavyatom'][...,None]

        concat_val_dict = {
            'seq_type': 2, 
            'res_nb': 0, 
            'fg_type': num_total_type, 
            'pos_heavyatom': 0, 
            'mask_heavyatom': True, 
            'fg_flag': True, 
            'fg_context_flag': True,
            'mask': True
            }
        pad_dict = {
            'seq_type': 0,
            'res_nb': 0,
            'fg_type': num_total_type, 
            'pos_heavyatom': 0, 
            'type_heavyatom': 0,
            'mask_heavyatom': False,
            'fg_flag': False,
            'fg_context_flag': False,
            'mask': False
        }
        
        concat_batch = {}
        
        for key, val in batch.items():
            if key in concat_val_dict.keys():
                if torch.is_tensor(val) and len(val.shape) > 1:
                    concat_val = []
                    if val.shape[1] == seq_len:
                        for i, one_sample in enumerate(val):

                            mask = batch['mask'][i]
                            one_sample_cond = one_sample[mask]
                            fg_num = fg_nums[i]
                            one_sample_gen = (torch.ones_like(one_sample_cond)[[0]]
                                            .repeat(fg_num, *(1,)*len(one_sample.shape[1:]))
                                            * concat_val_dict[key])
                            
                            one_sample_cat = torch.cat([one_sample_cond, one_sample_gen], dim=0)

                            concat_val.append(one_sample_cat)
                    else:
                        concat_val = val

                    concat_val = pad_sequence(concat_val, batch_first=True, padding_value=pad_dict[key])

            else:
                concat_val = val
            
            concat_batch[key] = concat_val
        
        return concat_batch
    
    @torch.no_grad()
    def assign_com_and_joint_size_with_translate(self, batch, pregen_model_fg, pregen_model_linker):
        batch_size, seq_len = batch['seq_type'].shape

        if pregen_model_fg is not None:
            fg_nums = pregen_model_fg(size=batch_size, batch=batch)
            coms = batch['cond_com']

        else:
            (coms, fg_nums ) = (
                batch['cond_com'], batch['fg_size']
            )
        
        if pregen_model_linker is not None:
            linker_nums = pregen_model_linker(size=batch_size, batch=batch)
            coms = batch['cond_com']

        else:
            (coms, linker_nums ) = (
                batch['cond_com'], batch['linker_size']
            )

        batch['gen_com'], batch['fg_size'], batch['linker_size'] = (
            coms, fg_nums, linker_nums
        )

        batch['pos_heavyatom'] -= batch['gen_com'][:,None,None,:]
        batch['pos_heavyatom'] *= batch['mask_heavyatom'][...,None]

        concat_val_dict = {
            'seq_type': 2, 
            'res_nb': 0, 
            'fg_type': num_total_type, 
            'pos_heavyatom': 0, 
            'mask_heavyatom': True, 
            'fg_flag': True, 
            'fg_context_flag': True,
            'mask': True
            }
        pad_dict = {
            'seq_type': 0,
            'res_nb': 0,
            'fg_type': num_total_type, 
            'pos_heavyatom': 0, 
            'type_heavyatom': 0,
            'mask_heavyatom': False,
            'fg_flag': False,
            'fg_context_flag': False,
            'mask': False
        }
        
        concat_batch = {}
        gen_nums = fg_nums + linker_nums

        for key, val in batch.items():
            if key in concat_val_dict.keys():
                if torch.is_tensor(val) and len(val.shape) > 1:
                    concat_val = []
                    if val.shape[1] == seq_len:
                        for i, one_sample in enumerate(val):
                            mask = batch['mask'][i]
                            one_sample_cond = one_sample[mask]
                            gen_num = gen_nums[i]
                            one_sample_gen = (torch.ones_like(one_sample_cond)[[0]]
                                            .repeat(gen_num, *(1,)*len(one_sample.shape[1:]))
                                            * concat_val_dict[key])
                            
                            one_sample_cat = torch.cat([one_sample_cond, one_sample_gen], dim=0)

                            concat_val.append(one_sample_cat)

                    else:
                        concat_val = val

                    concat_val = pad_sequence(concat_val, batch_first=True, padding_value=pad_dict[key])

            else:
                concat_val = val
            
            concat_batch[key] = concat_val

        concat_wo_linker = []
        val = batch['fg_context_flag']
        for i, one_sample in enumerate(val):
            mask = batch['mask'][i]
            one_sample_cond = one_sample[mask]
            fg_num = fg_nums[i]
            linker_num = linker_nums[i]
            one_sample_gen = torch.cat([
                                (torch.ones_like(one_sample_cond)[[0]]
                                .repeat(fg_num, *(1,)*len(one_sample.shape[1:]))
                                * True),
                                (torch.ones_like(one_sample_cond)[[0]]
                                .repeat(linker_num, *(1,)*len(one_sample.shape[1:]))
                                * False),
                            ])
            
            one_sample_cat = torch.cat([one_sample_cond, one_sample_gen], dim=0)

            concat_wo_linker.append(one_sample_cat)
        concat_wo_linker = pad_sequence(concat_wo_linker, batch_first=True, padding_value=False)

        concat_batch['wo_linker_flag'] = concat_wo_linker
        
        return concat_batch

    @torch.no_grad()
    def generate_linker_context(self, v, pos, fg, mask_fg, 
                                linker_size=None, pregen_model_linker=None):
        (batch_atom_pos, 
         batch_atom_type, 
         batch_context_mask, 
         batch_linker_mask, 
         batch_is_aromatic) = (
            [], [], [], [], []
            )
        
        linker_center = torch.tensor([0, 0, 0]).to(pos)

        fg_atom_types, fg_atom_poses, fg_atom_aromatic = [], [], [] 
        batch_size = pos.shape[0]

        for current_v, current_pos, current_fg, current_mask in zip(v, pos, fg, mask_fg):
            atom_type, atom_pos, is_aromatic = self.decode_fg_to_atom(
                current_v[current_mask], current_pos[current_mask], current_fg[current_mask]
                )
            fg_atom_types.append(atom_type)
            fg_atom_poses.append(atom_pos)
            fg_atom_aromatic.append(is_aromatic)

            assert(len(atom_type) == len(atom_pos) == len(is_aromatic))

        if pregen_model_linker is not None:
            linker_size = pregen_model_linker(size=batch_size) 

        atom_pos -= linker_center[None, :]

        for current_size, atom_type, atom_pos, is_aromatic in zip(linker_size, 
                                                                  fg_atom_types, 
                                                                  fg_atom_poses, 
                                                                  fg_atom_aromatic):
            
            linker_pos = torch.zeros((current_size, 3)).to(atom_pos)
            linker_type = torch.ones((current_size)).to(atom_type) * num_total_type
            linker_aromatic = torch.zeros((current_size)).to(atom_type).bool() 

            atom_pos_cat = torch.cat([atom_pos, linker_pos], dim=0)
            atom_type_cat = torch.cat([atom_type, linker_type], dim=0)
            aromatic_cat = torch.cat([is_aromatic, linker_aromatic], dim=0)
            
            batch_atom_pos.append(atom_pos_cat)
            batch_atom_type.append(atom_type_cat)
            batch_is_aromatic.append(aromatic_cat)
            
            context_mask = torch.concat([
                torch.ones(len(atom_type)), torch.ones(current_size)
                ]).to(atom_type).bool()
            linker_mask = torch.concat([
                torch.zeros(len(atom_type)), torch.ones(current_size)
                ]).to(atom_type).bool()
            
            batch_context_mask.append(context_mask)
            batch_linker_mask.append(linker_mask)

            assert(
                len(atom_pos_cat) == len(atom_type_cat) 
                == len(aromatic_cat) == len(context_mask) 
                == len(linker_mask)
                )

        batch_atom_pos = pad_sequence(batch_atom_pos, batch_first=True).float()
        batch_atom_type = pad_sequence(batch_atom_type, batch_first=True, padding_value=num_total_type).long()
        batch_context_mask = pad_sequence(batch_context_mask, batch_first=True).bool()
        batch_linker_mask = pad_sequence(batch_linker_mask, batch_first=True).bool()
        batch_is_aromatic = pad_sequence(batch_is_aromatic, batch_first=True).bool()

        assert(
                batch_atom_pos.shape[1] == batch_atom_type.shape[1] 
                == batch_context_mask.shape[1] == batch_linker_mask.shape[1] 
                == batch_is_aromatic.shape[1] 
                )

        return batch_atom_pos, batch_atom_type, batch_linker_mask, batch_context_mask, batch_is_aromatic
    
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
    def sample(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_type': True,
        },
        pregen_model = None
    ):  
        if pregen_model is not None:
            pregen_model_linker = pregen_model['pregen_linker']
            pregen_model_fg = pregen_model['pregen_fg']
        
        if self.scheme == 'two_stage':

            batch = self.assign_com_and_fgsize_with_translate(
                batch, pregen_model_fg
                )

            mask_fg = batch['fg_flag']
            mask_fg_context = batch['fg_context_flag']
            
            res_feat, pair_feat, R_0, p_0 = self.encode(
                batch,
                remove_structure = sample_opt.get('sample_structure', True),
                remove_type = sample_opt.get('sample_type', True)
            )
            v_0 = rotation_to_so3vec(R_0)
            s_0 = batch['fg_type']

            (final_v_fg, final_pos_fg, final_s_fg), traj_fg = self.fg_diffusion.sample(
                v_0, p_0, s_0, res_feat, pair_feat, 
                mask_fg, mask_fg_context, **sample_opt
                )
            
            (p_0_linker, s_0_linker, mask_linker, 
            mask_linker_context, batch_is_aromatic) = self.generate_linker_context(
                final_v_fg, final_pos_fg, final_s_fg, 
                mask_fg, pregen_model_linker=pregen_model_linker
                )
            
            (final_pos_linker, final_s_linker), traj_linker = self.linker_diffusion.sample(
                p_0_linker, s_0_linker, mask_linker, mask_linker_context, **sample_opt
                )
            
            batch['mol_mask'] = mask_linker_context
            batch['is_aromatic'] = batch_is_aromatic
            (final_pos, final_s) = (final_pos_linker, final_s_linker)
            traj = {'fg':traj_fg, 'linker':traj_linker}
        
        elif self.scheme == 'joint':
            batch = self.assign_com_and_joint_size_with_translate(
                batch, pregen_model_fg, pregen_model_linker
                )
            mask_fg = batch['fg_flag']
            mask_fg_context = batch['fg_context_flag']
            mask_wo_linker = batch['wo_linker_flag']
            
            res_feat, pair_feat, R_0, p_0 = self.encode(
                batch,
                remove_structure = sample_opt.get('sample_structure', True),
                remove_type = sample_opt.get('sample_type', True)
            )
            v_0 = rotation_to_so3vec(R_0)
            s_0 = batch['fg_type']

            (final_v_fg, final_pos_fg, final_s_fg), traj_fg = self.fg_diffusion.sample(
                v_0, p_0, s_0, res_feat, 
                pair_feat, mask_fg, mask_fg_context, 
                mask_wo_v=mask_wo_linker, **sample_opt
                )
            final_pos, final_s, atom_mask, is_aromatic = self.decode_batch_fg_to_atom(
                final_v_fg, final_pos_fg, final_s_fg, mask_fg
                )
            batch['mol_mask'] = atom_mask
            batch['is_aromatic'] = is_aromatic
            (final_pos, final_s) = (final_pos, final_s + num_aa_types)
            traj = {'fg':traj_fg}

        final_pos = self.translate_back(final_pos, batch['gen_com'], batch['mol_mask'])
        return batch, (final_pos, final_s), traj
