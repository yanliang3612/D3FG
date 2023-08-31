from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from openbabel import openbabel as ob
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np
from rdkit.Chem import AllChem as Chem
UPGRADE_BOND_ORDER = {
    Chem.BondType.SINGLE:Chem.BondType.DOUBLE, 
    Chem.BondType.DOUBLE:Chem.BondType.TRIPLE, 
    Chem.BondType.AROMATIC:Chem.BondType.AROMATIC
    }
VAL_BOND_DICT = {1:Chem.BondType.SINGLE,
                 2:Chem.BondType.DOUBLE,
                 3:Chem.BondType.TRIPLE
                 }
from rdkit import Geometry
import os

def combine_fgs_to_rdmol(fg_smiles, fg_atom_pos, return_aromatic=False):
    is_aromatic = []
    for i, fg_type in enumerate(fg_smiles):
        rd_fg = Chem.MolFromSmiles(fg_type)

        n_atoms = len(rd_fg.GetAtoms())
        rd_conf = Chem.Conformer(n_atoms)

        for atom in rd_fg.GetAtoms():
            j = atom.GetIdx()
            pos = fg_atom_pos[i][j]
            rd_coords = Geometry.Point3D(*pos)
            rd_conf.SetAtomPosition(j, rd_coords)

        rd_fg.AddConformer(rd_conf)
        
        for id, atom in enumerate(rd_fg.GetAtoms()):
            aromatic = rd_fg.GetAtomWithIdx(id).GetIsAromatic()
            is_aromatic.append(aromatic)
        
        if i == 0:
            rd_mol = rd_fg
        else:
            rd_mol = Chem.CombineMols(rd_mol, rd_fg)
    if return_aromatic:
        return rd_mol, is_aromatic
    return rd_mol

def rd_mol_to_ob_mol(rd_mol, remove_bond=False):
    '''
    Convert an RWMol to an OBMol, copying
    over the elements, coordinates, formal
    charges, bonds and aromaticity.
    '''
    ob_mol = ob.OBMol()
    ob_mol.BeginModify()
    rd_conf = rd_mol.GetConformer(0)

    for idx, rd_atom in enumerate(rd_mol.GetAtoms()):

        ob_atom = ob_mol.NewAtom()
        ob_atom.SetAtomicNum(rd_atom.GetAtomicNum())
        ob_atom.SetFormalCharge(rd_atom.GetFormalCharge())
        ob_atom.SetAromatic(rd_atom.GetIsAromatic())
        ob_atom.SetImplicitHCount(rd_atom.GetNumExplicitHs())

        rd_coords = rd_conf.GetAtomPosition(idx)
        ob_atom.SetVector(rd_coords.x, rd_coords.y, rd_coords.z)
    if remove_bond:
        ob_mol.EndModify()
        return ob_mol

    for rd_bond in rd_mol.GetBonds():

        # OB uses 1-indexing, rdkit uses 0
        i = rd_bond.GetBeginAtomIdx() + 1
        j = rd_bond.GetEndAtomIdx() + 1

        bond_type = rd_bond.GetBondType()
        if bond_type == Chem.BondType.SINGLE:
            bond_order = 1
        elif bond_type == Chem.BondType.DOUBLE:
            bond_order = 2
        elif bond_type == Chem.BondType.TRIPLE:
            bond_order = 3
        else:
            raise Exception('unknown bond type {}'.format(bond_type))


        ob_mol.AddBond(i, j, bond_order)
        ob_bond = ob_mol.GetBond(i, j)
        ob_bond.SetAromatic(rd_bond.GetIsAromatic())

    ob_mol.EndModify()
    return ob_mol

def count_nbrs_of_elem(atom, atomic_num):
    '''
    Count the number of neighbors atoms
    of atom with the given atomic_num.
    '''
    count = 0
    for nbr in ob.OBAtomAtomIter(atom):
        if nbr.GetAtomicNum() == atomic_num:
            count += 1
    return count

def connect_the_dots(mol, atoms, is_aromatic, maxbond=4):
    '''Custom implementation of ConnectTheDots.  This is similar to
    OpenBabel's version, but is more willing to make long bonds 
    (up to maxbond long) to keep the molecule connected.  It also 
    attempts to respect atom type information from struct.
    atoms and struct need to correspond in their order
    Assumes no hydrogens or existing bonds.
    '''
    pt = Chem.GetPeriodicTable()

    if len(atoms) == 0:
        return

    mol.BeginModify()

    #just going to to do n^2 comparisons, can worry about efficiency later
    coords = np.array([(a.GetX(),a.GetY(),a.GetZ()) for a in atoms])
    dists = squareform(pdist(coords))
    # types = [struct.channels[t].name for t in struct.c]

    for (i,a) in enumerate(atoms):
        for (j,b) in enumerate(atoms):
            if a == b:
                break
            if dists[i,j] < 0.01:  #reduce from 0.4
                continue #don't bond too close atoms
            if dists[i,j] < maxbond:
                flag = 0
                if is_aromatic[i] and is_aromatic[j]:
                    # print('Aromatic', ATOM_FAMILIES_ID['Aromatic'], indicators[i])
                    flag = ob.OB_AROMATIC_BOND
                # if 'Aromatic' in types[i] and 'Aromatic' in types[j]:
                #     flag = ob.OB_AROMATIC_BOND
                mol.AddBond(a.GetIdx(),b.GetIdx(),1,flag)

    atom_maxb = {}
    for (i,a) in enumerate(atoms):
        #set max valance to the smallest max allowed by openbabel or rdkit
        #since we want the molecule to be valid for both (rdkit is usually lower)
        maxb = ob.GetMaxBonds(a.GetAtomicNum())
        maxb = min(maxb,pt.GetDefaultValence(a.GetAtomicNum())) 

        if a.GetAtomicNum() == 16: # sulfone check
            if count_nbrs_of_elem(a, 8) >= 2:
                maxb = 6

        # if indicators[i][ATOM_FAMILIES_ID['Donor']]:
        #     maxb -= 1 #leave room for hydrogen
        # if 'Donor' in types[i]:
        #     maxb -= 1 #leave room for hydrogen
        atom_maxb[a.GetIdx()] = maxb
    
    #remove any impossible bonds between halogens
    for bond in ob.OBMolBondIter(mol):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if atom_maxb[a1.GetIdx()] == 1 and atom_maxb[a2.GetIdx()] == 1:
            mol.DeleteBond(bond)

    def get_bond_info(biter):
        '''Return bonds sorted by their distortion'''
        bonds = [b for b in biter]
        binfo = []
        for bond in bonds:
            bdist = bond.GetLength()
            #compute how far away from optimal we are
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            ideal = ob.GetCovalentRad(a1.GetAtomicNum()) + ob.GetCovalentRad(a2.GetAtomicNum()) 
            stretch = bdist-ideal
            binfo.append((stretch,bdist,bond))
        binfo.sort(reverse=True, key=lambda t: t[:2]) #most stretched bonds first
        return binfo

    binfo = get_bond_info(ob.OBMolBondIter(mol))
    #now eliminate geometrically poor bonds
    for stretch,bdist,bond in binfo:
        #can we remove this bond without disconnecting the molecule?
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        #as long as we aren't disconnecting, let's remove things
        #that are excessively far away (0.45 from ConnectTheDots)
        #get bonds to be less than max allowed
        #also remove tight angles, because that is what ConnectTheDots does
        if stretch > 0.45 or forms_small_angle(a1,a2) or forms_small_angle(a2,a1):
            #don't fragment the molecule
            if not reachable(a1,a2):
                continue
            mol.DeleteBond(bond)

    mol.EndModify()

def reachable_r(a,b, seenbonds):
    '''Recursive helper.'''

    for nbr in ob.OBAtomAtomIter(a):
        bond = a.GetBond(nbr).GetIdx()
        if bond not in seenbonds:
            seenbonds.add(bond)
            if nbr == b:
                return True
            elif reachable_r(nbr,b,seenbonds):
                return True
    return False


def reachable(a,b):
    '''Return true if atom b is reachable from a without using the bond between them.'''
    if a.GetExplicitDegree() == 1 or b.GetExplicitDegree() == 1:
        return False #this is the _only_ bond for one atom
    #otherwise do recursive traversal
    seenbonds = set([a.GetBond(b).GetIdx()])
    return reachable_r(a,b,seenbonds)

def forms_small_angle(a,b,cutoff=45):
    '''Return true if bond between a and b is part of a small angle
    with a neighbor of a only.'''

    for nbr in ob.OBAtomAtomIter(a):
        if nbr != b:
            degrees = b.GetAngle(a,nbr)
            if degrees < cutoff:
                return True
    return False

def calc_valence(rdatom):
    '''Can call GetExplicitValence before sanitize, but need to
    know this to fix up the molecule to prevent sanitization failures'''
    cnt = 0.0
    for bond in rdatom.GetBonds():
        cnt += bond.GetBondTypeAsDouble()
    return cnt

def convert_ob_mol_to_rd_mol(ob_mol):
    '''Convert OBMol to RDKit mol, fixing up issues'''
    ob_mol.DeleteHydrogens()
    n_atoms = ob_mol.NumAtoms()
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    for ob_atom in ob.OBMolAtomIter(ob_mol):
        rd_atom = Chem.Atom(ob_atom.GetAtomicNum())
        #TODO copy format charge
        if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:
            #don't commit to being aromatic unless rdkit will be okay with the ring status
            #(this can happen if the atoms aren't fit well enough)
            rd_atom.SetIsAromatic(True)
        i = rd_mol.AddAtom(rd_atom)
        ob_coords = ob_atom.GetVector()
        x = ob_coords.GetX()
        y = ob_coords.GetY()
        z = ob_coords.GetZ()
        rd_coords = Geometry.Point3D(x, y, z)
        rd_conf.SetAtomPosition(i, rd_coords)

    rd_mol.AddConformer(rd_conf)

    for ob_bond in ob.OBMolBondIter(ob_mol):
        i = ob_bond.GetBeginAtomIdx()-1
        j = ob_bond.GetEndAtomIdx()-1
        bond_order = ob_bond.GetBondOrder()
        if bond_order == 1:
            rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
        elif bond_order == 2:
            rd_mol.AddBond(i, j, Chem.BondType.DOUBLE)
        elif bond_order == 3:
            rd_mol.AddBond(i, j, Chem.BondType.TRIPLE)
        else:
            raise Exception('unknown bond order {}'.format(bond_order))

        if ob_bond.IsAromatic():
            bond = rd_mol.GetBondBetweenAtoms (i,j)
            bond.SetIsAromatic(True)

    # Chem.MMFFOptimizeMolecule(rd_mol)

    rd_mol = Chem.RemoveHs(rd_mol, sanitize=False)

    pt = Chem.GetPeriodicTable()
    #if double/triple bonds are connected to hypervalent atoms, decrement the order

    positions = rd_mol.GetConformer().GetPositions()
    nonsingles = []
    for bond in rd_mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE or bond.GetBondType() == Chem.BondType.TRIPLE:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            dist = np.linalg.norm(positions[i]-positions[j])
            nonsingles.append((dist,bond))
    nonsingles.sort(reverse=True, key=lambda t: t[0])

    for (d,bond) in nonsingles:
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if calc_valence(a1) > pt.GetDefaultValence(a1.GetAtomicNum()) or \
           calc_valence(a2) > pt.GetDefaultValence(a2.GetAtomicNum()):
            btype = Chem.BondType.SINGLE
            if bond.GetBondType() == Chem.BondType.TRIPLE:
                btype = Chem.BondType.DOUBLE
            bond.SetBondType(btype)

    for atom in rd_mol.GetAtoms():
        #set nitrogens with 4 neighbors to have a charge
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
            atom.SetFormalCharge(1)

    rd_mol = Chem.AddHs(rd_mol,addCoords=True)

    positions = rd_mol.GetConformer().GetPositions()
    center = np.mean(positions[np.all(np.isfinite(positions),axis=1)],axis=0)
    for atom in rd_mol.GetAtoms():
        i = atom.GetIdx()
        pos = positions[i]
        if not np.all(np.isfinite(pos)):
            #hydrogens on C fragment get set to nan (shouldn't, but they do)
            rd_mol.GetConformer().SetAtomPosition(i,center)

    # try:
    #     Chem.SanitizeMol(rd_mol,Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
    # except:
    #     print ('MolReconsError')
    # try:
    #     Chem.SanitizeMol(rd_mol,Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
    # except: # mtr22 - don't assume mols will pass this
    #     pass
    #     # dkoes - but we want to make failures as rare as possible and should debug them
    #     m = pybel.Molecule(ob_mol)
    #     i = np.random.randint(1000000)
    #     outname = 'bad%d.sdf'%i
    #     print("WRITING",outname)
    #     m.write('sdf',outname,overwrite=True)
    #     pickle.dump(struct,open('bad%d.pkl'%i,'wb'))

    #but at some point stop trying to enforce our aromaticity -
    #openbabel and rdkit have different aromaticity models so they
    #won't always agree.  Remove any aromatic bonds to non-aromatic atoms
    for bond in rd_mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if bond.GetIsAromatic():
            if not a1.GetIsAromatic() or not a2.GetIsAromatic():
                bond.SetIsAromatic(False)
        elif a1.GetIsAromatic() and a2.GetIsAromatic():
            bond.SetIsAromatic(True)

    return rd_mol


def fixup(atoms, mol, is_aromatic):
    '''Set atom properties to match channel.  Keep doing this
    to beat openbabel over the head with what we want to happen.'''

    mol.SetAromaticPerceived(True)  #avoid perception
    for i, atom in enumerate(atoms):
        # ch = struct.channels[t]
        ind = is_aromatic[i]

        if ind:
            atom.SetAromatic(True)
            atom.SetHyb(2)

        # if ind[ATOM_FAMILIES_ID['Donor']]:
        #     if atom.GetExplicitDegree() == atom.GetHvyDegree():
        #         if atom.GetHvyDegree() == 1 and atom.GetAtomicNum() == 7:
        #             atom.SetImplicitHCount(2)
        #         else:
        #             atom.SetImplicitHCount(1) 


        # elif ind[ATOM_FAMILIES_ID['Acceptor']]: # NOT AcceptorDonor because of else
        #     atom.SetImplicitHCount(0)   

        if (atom.GetAtomicNum() in (7, 8)) and atom.IsInRing():     # Nitrogen, Oxygen
            #this is a little iffy, ommitting until there is more evidence it is a net positive
            #we don't have aromatic types for nitrogen, but if it
            #is in a ring with aromatic carbon mark it aromatic as well
            acnt = 0
            for nbr in ob.OBAtomAtomIter(atom):
                if nbr.IsAromatic():
                    acnt += 1
            if acnt > 1:
                atom.SetAromatic(True)


def postprocess_rd_mol_1(rd_mol):

    rd_mol = Chem.RemoveHs(rd_mol)

    # Construct bond nbh list
    nbh_list = {}
    for bond in rd_mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx() 
        if begin not in nbh_list: nbh_list[begin] = [end]
        else: nbh_list[begin].append(end)
            
        if end not in nbh_list: nbh_list[end] = [begin]
        else: nbh_list[end].append(begin)

    # Fix missing bond-order
    for atom in rd_mol.GetAtoms():
        idx = atom.GetIdx()
        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            for j in nbh_list[idx]:
                if j <= idx: continue
                nb_atom = rd_mol.GetAtomWithIdx(j)
                nb_radical = nb_atom.GetNumRadicalElectrons()
                if nb_radical > 0:
                    bond = rd_mol.GetBondBetweenAtoms(idx, j)
                    bond.SetBondType(UPGRADE_BOND_ORDER[bond.GetBondType()])
                    nb_atom.SetNumRadicalElectrons(nb_radical - 1)
                    num_radical -= 1
            atom.SetNumRadicalElectrons(num_radical)

        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            atom.SetNumRadicalElectrons(0)
            num_hs = atom.GetNumExplicitHs()
            atom.SetNumExplicitHs(num_hs + num_radical)
            
    return rd_mol


def postprocess_rd_mol_2(rd_mol):
    rd_mol_edit = Chem.RWMol(rd_mol)

    ring_info = rd_mol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]
    for i, ring_a in enumerate(rings):
        if len(ring_a) == 3:
            non_carbon = []
            atom_by_symb = {}
            for atom_idx in ring_a:
                symb = rd_mol.GetAtomWithIdx(atom_idx).GetSymbol()
                if symb != 'C':
                    non_carbon.append(atom_idx)
                if symb not in atom_by_symb:
                    atom_by_symb[symb] = [atom_idx]
                else:
                    atom_by_symb[symb].append(atom_idx)
            if len(non_carbon) == 2:
                rd_mol_edit.RemoveBond(*non_carbon)
            if 'O' in atom_by_symb and len(atom_by_symb['O']) == 2:
                rd_mol_edit.RemoveBond(*atom_by_symb['O'])
                rd_mol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).SetNumExplicitHs(
                    rd_mol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).GetNumExplicitHs() + 1
                )
                rd_mol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).SetNumExplicitHs(
                    rd_mol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).GetNumExplicitHs() + 1
                )
    rd_mol = rd_mol_edit.GetMol()

    for atom in rd_mol.GetAtoms():
        if atom.GetFormalCharge() > 0:
            atom.SetFormalCharge(0)

    return rd_mol


def reconstruct_from_generated(ob_mol, is_aromatic):
    mol = ob_mol
    atoms = []
    for ob_atom in ob.OBMolAtomIter(ob_mol):
        atoms.append(ob_atom)

    fixup(atoms, mol, is_aromatic)

    connect_the_dots(mol, atoms, is_aromatic, 2)
    fixup(atoms, mol, is_aromatic)
    
    mol.AddPolarHydrogens()
    mol.PerceiveBondOrders()
    fixup(atoms, mol, is_aromatic)

    for (i,a) in enumerate(atoms):
        ob.OBAtomAssignTypicalImplicitHydrogens(a)
    fixup(atoms, mol, is_aromatic)

    mol.AddHydrogens()
    fixup(atoms, mol, is_aromatic)

    #make rings all aromatic if majority of carbons are aromatic
    for ring in ob.OBMolRingIter(mol):
        if 5 <= ring.Size() <= 6:
            carbon_cnt = 0
            aromatic_ccnt = 0
            for ai in ring._path:
                a = mol.GetAtom(ai)
                if a.GetAtomicNum() == 6:
                    carbon_cnt += 1
                    if a.IsAromatic():
                        aromatic_ccnt += 1
            if aromatic_ccnt >= carbon_cnt/2 and aromatic_ccnt != ring.Size():
                #set all ring atoms to be aromatic
                for ai in ring._path:
                    a = mol.GetAtom(ai)
                    a.SetAromatic(True)

    #bonds must be marked aromatic for smiles to match
    for bond in ob.OBMolBondIter(mol):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.IsAromatic() and a2.IsAromatic():
            bond.SetAromatic(True)
            
    mol.PerceiveBondOrders()

    rd_mol = convert_ob_mol_to_rd_mol(mol)

    # Post-processing
    rd_mol = postprocess_rd_mol_1(rd_mol)
    rd_mol = postprocess_rd_mol_2(rd_mol)

    return rd_mol

def save_valid_mol(sdf_dir, receptor_name, ith, rd_mol):
    receptor_dir = sdf_dir + "/" + os.path.dirname(receptor_name)
    if not os.path.exists(receptor_dir):
        os.makedirs(receptor_dir)
    w = Chem.SDWriter(
        sdf_dir
        + "/{}_valid{}.sdf".format(
            receptor_name[:-4],
            ith,
        )
    )
    w.write(rd_mol)
    return sdf_dir + "/{}_valid{}.sdf".format(receptor_name[:-4], ith,)
                        