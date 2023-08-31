from utils.evaluation import scoring_func
from utils.evaluation.docking_vina import VinaDockingTask, GninaDockingTask
from multiprocessing import Pool
import argparse
import os
from functools import partial
from glob import glob
from tqdm.auto import tqdm
from copy import deepcopy
import numpy as np
from rdkit import RDLogger
from pathlib import Path
from rdkit import Chem
import torch


def eval_single(
        ligand_filename, 
        protein_filename, 
        reference_filename, 
        use_ref_center, 
        save_pose=True,
        docking_modes = []
        ):
    mol = Chem.SDMolSupplier(ligand_filename)[0]
    mol_ref = Chem.SDMolSupplier(reference_filename)[0]
    chem_results = scoring_func.get_chem(mol)
    center = None

    if use_ref_center:
        pos = mol_ref.GetConformer(0).GetPositions()
        center = np.array(pos).mean(axis=0).tolist()


    vina_task = VinaDockingTask.from_generated_mol(ligand_filename,
                                                protein_path=protein_filename, **{'center': center})
    if 'vina_score' in docking_modes:
        score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness, save_pose=save_pose)
    else:
        score_only_results = None
    
    if 'vina_minimize' in docking_modes:
        minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness, save_pose=save_pose)
    else:
        minimize_results = None
    
    if 'vina_dock' in docking_modes:
        dock_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness, save_pose=save_pose)
    else:
        dock_results = None
    
    if 'gnina' in docking_modes:
        gnina_task = GninaDockingTask.from_generated_mol(ligand_filename,
                                                        protein_path=protein_filename, **{'center': center})
        cnn_dock_results = gnina_task.run(save_pose=True)
    else:
        cnn_dock_results = None

    ligand_name = os.path.basename(ligand_filename)[:-4]
    protein_name = os.path.dirname(ligand_filename).split('/')[-1]

    out_dict = {
        'entry':(protein_name + '/' + ligand_name), 
        'vina_results':minimize_results, 
        'dock_results':dock_results,
        'score_results':score_only_results,
        'cnn_dock_results': cnn_dock_results,
        'chem_results':chem_results
        }
    
    torch.save(out_dict, os.path.join(args.result_path, (protein_name + '/' + ligand_name) + '.pt'))
    return out_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ligand_path', default='/usr/commondata/public/conformation_generation/SbddBaselineSamples/d3fg_joint/', type=str)  # 'sampling_results/targetdiff_vina_docked.pt'
    parser.add_argument('--eval_num_examples', type=int, default=100)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--protein_path', type=str, default='/usr/commondata/public/conformation_generation/data/crossdocked_pocket10/')
    parser.add_argument('--docking_modes', type=str,
                        default=['gnina', 'vina_score', 'vina_minimize', 'vina_full'])
    parser.add_argument('--exhaustiveness', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--result_path', type=str, default='/usr/commondata/public/conformation_generation/SbddBaselineSamples/d3fg_joint/')
    parser.add_argument('--aggregate_meta', type=eval, default=False)
    parser.add_argument('--use_reference_center', type=eval, default=False)

    args = parser.parse_args()

    ligand_path = Path(args.ligand_path)
    protein_list = os.listdir(ligand_path)

    protein_path = Path(args.protein_path)

    meta_data = []

    for protein_name in protein_list:
        ligand_base_dir = ligand_path / protein_name
        ligand_list = os.listdir(ligand_base_dir)

        pocket_result = []
        
        protein_base_dir = protein_path / protein_name

        success_num = 0
        
        for i, ligand_name in enumerate(ligand_list):

            if success_num >= args.eval_num_examples:
                break

            try:

                ligand_filename = str(ligand_base_dir / ligand_name)
                
                pocket_name = '_'.join(ligand_name[:-4].split('_')[:-1]) + '_pocket10.pdb'
                protein_filename = str(protein_base_dir / pocket_name)

                reference_ligand_name = '_'.join(ligand_name.split('_')[:-1]) + '.sdf'
                reference_filename = str(protein_base_dir / reference_ligand_name)

                if i == 0 :
                    out_dict = eval_single(
                        reference_filename, 
                        protein_filename, 
                        reference_filename, 
                        args.use_reference_center,
                        save_pose=False,
                        docking_modes=args.docking_modes
                        )
                    pocket_result.append(out_dict)

                
            
                out_dict = eval_single(
                            ligand_filename, 
                            protein_filename, 
                            reference_filename, 
                            args.use_reference_center,
                            docking_modes=args.docking_modes
                            )
                
                pocket_result.append(out_dict)
                success_num += 1
                print('{}:, {}/{}:' .format(pocket_name,success_num,args.eval_num_examples))

                
            except:
                continue
        
        meta_data.append(pocket_result)
            

    torch.save(meta_data, os.path.join(args.result_path, 'meta_results.pt'))

    