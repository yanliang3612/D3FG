from models.utils.geometry import *
from models.utils.so3 import *
from datasets.molecule.constants import *
from datasets.protein.constants import *
import torch 
from rdkit import Geometry
from rdkit.Chem import AllChem as Chem
import torch
from .reconstruct import *


shift_class = num_aa_types


def reconstruct_from_gen_fg(current_pos, current_v, current_fg, remove_bond=True):
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

    for i in range(len(fg_smiles_raw)):
        R = so3vec_to_rotation(torch.tensor(current_v[i])).float().unsqueeze(dim=0)
        t = torch.tensor(current_pos[i]).float().unsqueeze(dim=0)
        p = torch.tensor(fg_local_pos[i]).float().unsqueeze(dim=0)
        global_fg_pos = local_to_global(R, t, p).tolist()[0]
        fg_atom_pos.append(global_fg_pos)
    
    rd_mol, is_aromatic = combine_fgs_to_rdmol(fg_smiles_raw, fg_atom_pos)
    ob_mol = rd_mol_to_ob_mol(rd_mol, remove_bond=remove_bond)
    rd_mol, smiles = reconstruct_from_generated(ob_mol, is_aromatic)

    return rd_mol, smiles
    