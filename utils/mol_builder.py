
from datasets.molecule.constants import *
from datasets.protein.constants import *
from models.utils.geometry import *
from models.utils.so3 import *
from rdkit import Geometry
import openbabel
import tempfile
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams
import warnings
import subprocess
import os
from collections import Counter

from .reconstruct import *
shift_class = num_aa_types

def write_xyz_file(coords, atom_types, filename):
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.mkdir(dir)
    filename = filename + '.xyz'
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, 'w') as f:
        f.write(out)
    return filename


def obabel_recover_bond(positions, atom_types):
    with tempfile.NamedTemporaryFile() as tmp:
        temp_dir = './experiments/temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        tmp_file = './experiments/temp' + tmp.name

        # Write xyz file
        xyz_file = write_xyz_file(positions, atom_types, tmp_file)
        sdf_file = tmp_file + '.sdf'
        # subprocess.run(f'obabel {xyz_file} -O {sdf_file}', shell=True)

        # Convert to sdf file with openbabel
        # openbabel will add bonds
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, xyz_file)

        obConversion.WriteFile(ob_mol, sdf_file)

        # Read sdf file with RDKit
        rd_mol = Chem.SDMolSupplier(sdf_file, sanitize=False)[0]
    return rd_mol

def refine_recover_bond(rd_mol, is_aromatic):
    try:
        is_aromatic = is_aromatic.tolist()
    except:
        is_aromatic = is_aromatic
    
    ob_mol = rd_mol_to_ob_mol(rd_mol)
    rd_mol = reconstruct_from_generated(ob_mol, is_aromatic)

    return rd_mol

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

def ring_type_from_mol(mol):
    ring_info = mol.GetRingInfo()
    ring_type = [len(r) for r in ring_info.AtomRings()]
    return ring_type

def clean_frags(mol, threshold=10, filter_ring=True):
    mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    if mol.GetNumAtoms() < threshold:
        return Chem.MolFromSmiles('C.C')
    if filter_ring:
        ring_type = Counter(ring_type_from_mol(mol))
        if 4 in ring_type:
            mol = Chem.MolFromSmiles('C.C')
        if 3 in ring_type and ring_type[3]>1:
            mol = Chem.MolFromSmiles('C.C')
    return mol


def build_mol(
        current_pos, 
        current_fg, 
        current_v=None, 
        process=True, 
        is_aromatic=None,
        uff_iter=500,
        clean=True, 
        threshold = 16
        ):
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
        
        if current_v is not None:
            R = so3vec_to_rotation(torch.tensor(current_v[i]).float()).unsqueeze(dim=0)
        else:
            R = so3vec_to_rotation(torch.tensor([0, 0, 0]).float()).unsqueeze(dim=0)

        t = torch.tensor(current_pos[i]).float().unsqueeze(dim=0)
        p = torch.tensor(fg_local_pos[i]).float().unsqueeze(dim=0)
        global_fg_pos = local_to_global(R, t, p).tolist()[0]
        fg_atom_pos.append(global_fg_pos)
    
    rd_mol = combine_fgs_to_rdmol(fg_smiles_raw, fg_atom_pos)
    positions = rd_mol.GetConformer(0).GetPositions()
    atom_types = [atom.GetSymbol() for atom in rd_mol.GetAtoms()]

    try:
        mol = refine_recover_bond(rd_mol, is_aromatic)
        if clean:
            mol = clean_frags(mol, threshold)
    except:
        mol = obabel_recover_bond(positions, atom_types)

    if "." in Chem.MolToSmiles(mol):
        mol = obabel_recover_bond(positions, atom_types)
    
    if clean:
        mol = clean_frags(mol, threshold)
    
    if process:
        mol = process_molecule(mol, relax_iter=uff_iter)

    return mol, Chem.MolToSmiles(mol)


def process_molecule(rdmol, add_hydrogens=False, sanitize=False, relax_iter=0,
                     largest_frag=False):
    """
    Apply filters to an RDKit molecule. Makes a copy first.
    Args:
        rdmol: rdkit molecule
        add_hydrogens
        sanitize
        relax_iter: maximum number of UFF optimization iterations
        largest_frag: filter out the largest fragment in a set of disjoint
            molecules
    Returns:
        RDKit molecule or None if it does not pass the filters
    """

    # Create a copy
    mol = Chem.Mol(rdmol)

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            warnings.warn('Sanitization failed. Returning None.')
            return None

    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=(len(mol.GetConformers()) > 0))

    if largest_frag:
        mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        if sanitize:
            # sanitize the updated molecule
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                return None

    if relax_iter > 0:
        if not UFFHasAllMoleculeParams(mol):
            warnings.warn('UFF parameters not available for all atoms. '
                          'Returning None.')
            return mol

        try:
            uff_relax(mol, relax_iter)
            if sanitize:
                # sanitize the updated molecule
                Chem.SanitizeMol(mol)
            return mol
        except (RuntimeError, ValueError) as e:
            return mol

    return mol

def uff_relax(mol, max_iter=200):
    """
    Uses RDKit's universal force field (UFF) implementation to optimize a
    molecule.
    """
    more_iterations_required = UFFOptimizeMolecule(mol, maxIters=max_iter)
    if more_iterations_required:
        warnings.warn(f'Maximum number of FF iterations reached. '
                      f'Returning molecule after {max_iter} relaxation steps.')
    return more_iterations_required
