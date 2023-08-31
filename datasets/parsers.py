from .protein.parsers import parse_biopython_structure
from .molecule.parsers import parse_fg_molecule
from .molecule.hm_parsers import parse_fg_hm_molecule

def parse_protein_ligand_pairs(task):
    protein_data = parse_biopython_structure(task)
    
    ligand_data = parse_fg_molecule(task)

    if (ligand_data is not None) and (protein_data is not None):
        parsed_pair = {'entry': (task['pdb_entry'], task['sdf_entry']), 'protein': protein_data, 'ligand': ligand_data}
    else:
        parsed_pair = None
    
    return parsed_pair

def parse_protein_ligand_pairs_with_hm(task):
    protein_data = parse_biopython_structure(task)
    
    ligand_data = parse_fg_hm_molecule(task)

    if (ligand_data is not None) and (protein_data is not None):
        parsed_pair = {'entry': (task['pdb_entry'], task['sdf_entry']), 'protein': protein_data, 'ligand': ligand_data}
    else:
        parsed_pair = None
    
    return parsed_pair