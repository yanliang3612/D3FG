import pickle
import os
from tqdm.auto import tqdm
import numpy as np
# import EFGs
from rdkit import Chem
# from EFGs import mol2frag
from rdkit.Chem import AllChem
from .frame_fg import *
from .constants import *
ALIGNED_RMSD = 0.3
pattern_order = r'[0-9]'
import pandas as pd
import re
failed_num = 0

def get_fg_pos_and_type(rdmol, idx):
    fg_pos = []
    atomic_nums = []
    atom_types = []
    c = rdmol.GetConformer()
    for atom_idx in idx:
        atomic_num = rdmol.GetAtomWithIdx(atom_idx).GetAtomicNum()
        atom_type = rdmol.GetAtomWithIdx(atom_idx).GetSymbol() 
        pos = c.GetAtomPosition(atom_idx)
        fg_pos.append([pos.x, pos.y, pos.z])
        atomic_nums.append(atomic_num)
        atom_types.append(atom_type)
    return fg_pos, np.array(atomic_nums), np.array(atom_types)

def prepare_single_atom(fg_smile, fg_pos):
    v = [0,0,0]
    type = fg_smile
    center_pos = fg_pos[0]
    frame_vec = v
    frame_pos = [[0,0,0],fg_pos[0]]
    return type, center_pos, frame_vec, frame_pos

def prepare_fg_atom(fg_smile, fg_pos):
    center, R, v, local_pos, framed_mol, rearrange_global_pos, idx_re = transform_into_fg_data(fg_smile, fg_pos)
    if fg_smile == 'NS(=O)=O':
        rmsd = Chem.rdMolAlign.CalcRMS(framed_mol, ref_nso2_c1)
        if rmsd <= ALIGNED_RMSD:
            fg_frame_vec = v
            fg_type = nso2_chirality1
            fg_center_pos = center
            fg_frame_pos= rearrange_global_pos
        else:
            fg_frame_vec = v
            fg_type = nso2_chirality2
            fg_center_pos = center
            fg_frame_pos = rearrange_global_pos

    elif fg_smile == 'O=CNO':
        rmsd = Chem.rdMolAlign.CalcRMS(framed_mol, ref_ocno_c1)
        if rmsd <= ALIGNED_RMSD:
            fg_frame_vec = v
            fg_type = ocno_chirality1
            fg_center_pos = center
            fg_frame_pos = rearrange_global_pos

        else:
            fg_frame_vec = v
            fg_type = ocno_chirality2
            fg_center_pos = center     
            fg_frame_pos = rearrange_global_pos
    
    else:   
        fg_frame_vec = v
        fg_type = fg_smile
        fg_center_pos = center
        fg_frame_pos = rearrange_global_pos        
    return fg_frame_vec, fg_type, fg_center_pos, fg_frame_pos

def conf_with_smiles(smiles, positions):
    mol = Chem.MolFromSmiles(smiles)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (positions[i][0], positions[i][1], positions[i][2]))
    mol.AddConformer(conf)
    return mol

def prepare_linker_atom(fg_smile, fg_pos):
    rdmol = conf_with_smiles(fg_smile, fg_pos)
    positions = rdmol.GetConformer(0).GetPositions()
    atom_types = [atom.GetSymbol() for atom in rdmol.GetAtoms()]

    return positions, atom_types


def parse_fg_hm_molecule(task, size_threshold=60):
    ligand_path = task['sdf_path']
    ligand_score = task['hs_path']
    force_preserve = task['force_preserve']
    try:
        rdmol = Chem.MolFromMolFile(ligand_path)
        c = rdmol.GetConformer()
        score = pd.read_csv(ligand_score, index_col=False)
        atom_hms = np.array([re.sub(pattern_order, '', sympol) for sympol in score['Atom Identifier'].values])
        score_hms = score['Fragment Hotspot Score'].values
    except:
        return None
    
    fg, single_c, fg_idx, single_c_idx = mol2frag(rdmol, returnidx=True)

    fg = fg + single_c
    fg_idx = fg_idx + single_c_idx

    fg_center_pos = []
    v_fgs = []
    type_fgs = []
    pos_fgs = []
    center_fgs = []
    score_fgs = []

    pos_linker = []
    mask_linker = []
    type_linker = []
    score_linker = []
    fg_idxes = []
    for fg_smile, idx in zip(fg, fg_idx):  
        
        atom_hm = atom_hms[np.array(idx)]
        score_hm = score_hms[np.array(idx)]

        fg_pos, atomic_num, atom_type = get_fg_pos_and_type(rdmol, idx)

        assert np.prod(atom_hm == atom_type), 'the score order is not aligned.'

        if fg_smile in merge_c:
            fg_smile = 'C'
        
        if (fg_smile in fragment_factory):
            fg_frame_vec, fg_type, fg_center_pos, fg_frame_pos = prepare_fg_atom(fg_smile, fg_pos)
            v_fgs.append(fg_frame_vec)
            type_fgs.append(fg_type)
            center_fgs.append(fg_center_pos)
            pos_fgs.append(fg_frame_pos)
            score_fgs.append([score_hm.mean()])
            fg_idxes.append(idx)

        atom_pos, atom_type = prepare_linker_atom(fg_smile, fg_pos)

        if fg_smile in fragment_factory:
            fg_mask = np.zeros_like(idx)
        else:
            fg_mask = np.ones_like(idx)
        
        mask_linker.append(fg_mask)
        pos_linker.append(atom_pos)
        type_linker.append(atom_type)
        score_linker.append(score_hm)

    mask_linker = np.concatenate(mask_linker)
    pos_linker = np.concatenate(pos_linker, axis=0)
    type_linker = np.concatenate(type_linker)
    score_linker = np.concatenate(score_linker)

    pos_fg_pad = []
    mask_fg_pad = []

    for pos in pos_fgs:
        fg_atom_num = len(pos)

        pos_pad = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float)
        mask_pad = torch.zeros([max_num_heavyatoms, ], dtype=torch.bool)
    
        pos_pad[:fg_atom_num] = torch.tensor(pos)
        mask_pad[:fg_atom_num] = True  

        pos_fg_pad.append(pos_pad)
        mask_fg_pad.append(mask_pad)

    fg_type_encode = [fg2class_dict[fg_smile] for fg_smile in type_fgs]
    atom_type_encode = [fg2class_dict[atom_smile] if atom_smile in single_atom else fg2class_dict['Others']
                         for atom_smile in type_linker]

    assert (
        len(center_fgs) 
        == len(pos_fg_pad) 
        == len(mask_fg_pad)
        == len(score_fgs)
        == len(fg_idxes)
        )

    assert(
        len(atom_type_encode)
        == len(pos_linker)
        == len(mask_linker)
        == len(score_linker)
    )

    other_ratio =  len([fg_smile for fg_smile in type_fgs if fg_smile == 'Others'])/len(type_fgs) if len(type_fgs) > 0 else 1.0

    if other_ratio > 0.5 and (not force_preserve):
        return None
    
    mol_size = len(atom_type_encode)

    if (mol_size > size_threshold) and (not force_preserve):
        print(' An extreme huge molecule has been detected: {}, drop it!'.format(str(Chem.MolToSmiles(rdmol))))
        return None
    
    if len(center_fgs) == 0:
        return  {
        'center_pos': torch.tensor(center_fgs).float(), 
        'pos_heavyatom': torch.tensor(pos_fg_pad).float(),
        'mask_heavyatom': torch.tensor(mask_fg_pad).bool(),
        'type_fg': torch.tensor(fg_type_encode).long(), 
        'v_fg': torch.tensor(v_fgs).bool(),
        'score_fg': torch.tensor(score_fgs).float(),
        'pos_linker': torch.from_numpy(pos_linker).float(),
        'mask_linker': torch.from_numpy(mask_linker).bool(),
        'type_linker': torch.tensor(atom_type_encode).long(),
        'score_linker': torch.tensor(score_linker).float(),
        'rd_mol': rdmol,
        'fg_idx': fg_idxes
        }

    return  {
        'center_pos': torch.tensor(center_fgs).float(), 
        'pos_heavyatom': torch.stack(pos_fg_pad, dim=0).float(),
        'mask_heavyatom': torch.stack(mask_fg_pad, dim=0).bool(),
        'type_fg': torch.tensor(fg_type_encode).long(), 
        'v_fg': torch.tensor(v_fgs).bool(),
        'score_fg': torch.tensor(score_fgs).float(),
        'pos_linker': torch.from_numpy(pos_linker).float(),
        'mask_linker': torch.from_numpy(mask_linker).bool(),
        'type_linker': torch.tensor(atom_type_encode).long(),
        'score_linker': torch.tensor(score_linker).float(),
        'rd_mol': rdmol,
        'fg_idx': fg_idxes
        }
        

