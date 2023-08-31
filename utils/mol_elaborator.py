import pickle
import os
from tqdm.auto import tqdm
import numpy as np
import EFGs
from rdkit import Chem
from EFGs import mol2frag
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
import rdkit
from datasets.molecule.parsers import get_fg_pos_and_type
from datasets.molecule.constants import *
from models.utils.so3 import *
from models.utils.geometry import *
from rdkit import Geometry
from .reconstruct import VAL_BOND_DICT
from .mol_builder import uff_relax

periodic_table = rdkit.Chem.GetPeriodicTable()

def decoder_to_fg_rdmol(new_pos, new_fg, new_v):
    fg_smiles = class2fg_dict[new_fg]
    fg_local_pos = motif_pos_fractory[fg_smiles]
    t = torch.tensor(new_pos).float().unsqueeze(dim=0)
    p = torch.tensor(fg_local_pos).float().unsqueeze(dim=0)
    R = so3vec_to_rotation(torch.tensor(new_v).float()).unsqueeze(dim=0)
    global_fg_pos = local_to_global(R, t, p).tolist()[0]
    rd_fg = Chem.MolFromSmiles(fg_smiles)
    n_atoms = len(rd_fg.GetAtoms())
    rd_conf = Chem.Conformer(n_atoms)
    for atom in rd_fg.GetAtoms():
        j = atom.GetIdx()
        pos = global_fg_pos[j]
        rd_coords = Geometry.Point3D(*pos)
        rd_conf.SetAtomPosition(j, rd_coords)
    rd_fg.AddConformer(rd_conf)

    return rd_fg

def elaborate_mol_1(sdf_path, rm_idx, new_pos, new_fg, new_v, uff_iter=500, process=True, use_old_center=True):
    new_pos = new_pos.flatten().tolist()
    new_fg = new_fg.flatten().tolist()[0]
    new_v = new_v.flatten().tolist()

    sdf_path = str(sdf_path)

    rd_mol_raw =  Chem.MolFromMolFile(sdf_path)
    if use_old_center:
        old_center = np.array(get_fg_pos_and_type(rd_mol_raw, rm_idx)[0]).mean(axis=1).tolist()
        new_pos = old_center
    
    mol, break_idx = remove_fg(rd_mol_raw, rm_idx)

    rd_fg = decoder_to_fg_rdmol(new_pos, new_fg, new_v)
    mol = add_fg(mol, rd_fg, break_idx)
    if process:
        mol = uff_relax(mol, max_iter=uff_iter)
    
    return mol




def get_fg_neighbor(rdmol, rm_idx):
    '''
    Find the fg's neighbor to remove bonds
    '''
    neighbors = []
    for atom_idx in rm_idx:
        neighbor = rdmol.GetAtomWithIdx(atom_idx).GetNeighbors()
        neighbors.append(neighbor)
    return neighbors

def cal_bond_dis(atom1, atom2):
    '''
    calculate the euclidean distance of 2 atoms
    input: RDGeom::Point3D
    '''
    return np.sqrt(np.sum(np.square(atom1-atom2)))

def remove_fg(rdmol, rm_idx):
    '''
    remove the fg from rdmol
    input:
        rdmol: the origin mol
        rm_idx: indices of atoms to remove
    '''
    neighbors = get_fg_neighbor(rdmol, rm_idx)
    mol = rdkit.Chem.rdchem.EditableMol(rdmol)
    mol.BeginBatchEdit()
    break_pos = []
    c = rdmol.GetConformer()
    for idx in range(len(rm_idx)):
        mol.RemoveAtom(rm_idx[idx]) #remove atom
        for neighbor in neighbors[idx]:
            mol.RemoveBond(rm_idx[idx], neighbor.GetIdx())#remove bond
            if neighbor.GetIdx() not in rm_idx:
                break_pos.append(c.GetAtomPosition(neighbor.GetIdx()))

    mol.CommitBatchEdit()
    mol = mol.GetMol()

    break_idx = []
    c = mol.GetConformer()
    for pos in break_pos: #update the idx by finding the same pos in new mol
        for idx in range(mol.GetNumAtoms()):
            atom = c.GetAtomPosition(idx)
            if np.sum(np.abs(pos - atom)) < 1e-5:
                break_idx.append(idx)
                break

    return mol, break_idx

def add_fg(rdmol, fg, break_idx):
    '''
    add a fg to rdmol, the bond is set to the closest atom from break_idx
    input:
        rdmol: the mol framework
        fg: the fg to be added
        break_idx: the broken bond's atom indices when remove the former fg
    '''
    add_idx = [] #index to add bond
    c_fg = fg.GetConformer()
    c_mol = rdmol.GetConformer()
    old_atom_num = rdmol.GetNumAtoms()
    bond_type_pairs = []
    for idx in break_idx:
        distance = []
        idxs = []
        bond_type_pair = []
        # for atom in fg.GetAtoms():
        #     print(atom.GetSymbol(), atom.GetExplicitValence(), atom.GetNoImplicit(), atom.GetNumExplicitHs(), atom.GetTotalNumHs())
        for new_idx in range(fg.GetNumAtoms()):
            atom = fg.GetAtomWithIdx(new_idx)
            permitted_val = periodic_table.GetNOuterElecs(atom.GetSymbol())
            explicit_val = atom.GetExplicitValence()

            if permitted_val - explicit_val > 0:
                break_atom = rdmol.GetAtomWithIdx(idx)
                permitted_val_break = periodic_table.GetNOuterElecs(break_atom.GetSymbol())
                explicit_val_break = break_atom.GetExplicitValence()
                dis = cal_bond_dis(c_mol.GetAtomPosition(idx), c_fg.GetAtomPosition(new_idx))
                distance.append(dis)
                idxs.append(new_idx)
                bond_type_pair.append((permitted_val - explicit_val, 
                                       permitted_val_break - explicit_val_break))
                
        res = list(zip(distance, idxs))
        res.sort()#find closest idx
        add_idx.append(res[0][1])# calculate the closest atom
        bond_type_pair = bond_type_pair[res[0][1]]
        bond_type_pairs.append(min(bond_type_pair))

    mol = rdkit.Chem.rdchem.EditableMol(rdmol)
    mol.BeginBatchEdit()

    for atom in fg.GetAtoms():
        mol.AddAtom(atom)

    for bond in fg.GetBonds():
        atom1 = bond.GetEndAtomIdx() 
        atom2 = bond.GetOtherAtomIdx(atom1) 
        mol.AddBond(atom1 + old_atom_num, atom2 + old_atom_num, bond.GetBondType())
    

    for i, idx in enumerate(add_idx):
        mol.AddBond(idx + old_atom_num, break_idx[i], VAL_BOND_DICT[bond_type_pairs[i]])

    mol.CommitBatchEdit()
    mol = mol.GetMol()
    c = mol.GetConformer()
    for idx in range(fg.GetNumAtoms()):
        c.SetAtomPosition(idx + old_atom_num,c_fg.GetAtomPosition(idx))
    return mol

if __name__ == '__main__':
    path = '/linhaitao/crossdocked_pocket10/BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3.sdf'

    rdmol =  Chem.MolFromMolFile(path)
    
    fg, single_c, fg_idx, single_c_idx = mol2frag(rdmol, returnidx=True)

    removed_idx = np.arange(0, rdmol.GetNumAtoms(), 1)
    removed_idx = np.setdiff1d(removed_idx, fg_idx[0])
    removed_idx = tuple(map(int, removed_idx))

    mol, break_idx = remove_fg(rdmol, fg_idx[0])
    removed_mol, removed_break_idx = remove_fg(rdmol, removed_idx)

    mol = add_fg(mol, removed_mol, break_idx)
    print(rdmol.GetNumBonds())
    print(mol.GetNumBonds())