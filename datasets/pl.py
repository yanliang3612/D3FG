from .dataset_loader import register_dataset
from torch.utils.data import Dataset
import torch
from pathlib import Path
import os
from tqdm.auto import tqdm
from .parsers import parse_protein_ligand_pairs
import logging
import joblib
import lmdb
import pickle

@register_dataset('crossdocked')
def get_crossdocked_dataset(cfg, transform):
    split_path_test = torch.load(cfg.split_path)['test']
    return PairedCrossDocked(
        raw_path = cfg.raw_path,
        data_path = cfg.get('data_path', None),
        preserve_pair = split_path_test,
        transform = transform,
    )

class PairedCrossDocked(Dataset):

    MAP_SIZE = 16*(1024*1024*1024) 
    def __init__(self, raw_path, data_path=None, preserve_pair=None, transform=None) -> None:
        super().__init__()

        self.transform = transform
        self.raw_path = Path(raw_path)
        self.index_path = os.path.join(self.raw_path, 'index.pkl')

        self.data_path = './data/' if data_path is None else data_path
        self.name2id_path = os.path.join(os.path.dirname(self.data_path), 'crossdocked_name2id.pt')
        self.processed_paired_path = os.path.join(os.path.dirname(self.data_path), 'crossdocked_processed.lmdb')
        self.db = None
        
        self.preserve_pair = preserve_pair
        if not os.path.exists(self.processed_paired_path):
            self._preprocess_paired()
        if not os.path.exists(self.name2id_path):
            self._precompute_name2id()
        self.name2id = torch.load(self.name2id_path)
        
    
    def _preprocess_paired(self):
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        preserve_name = list(sum(self.preserve_pair, ()))
        
        tasks = []
        process_bar = tqdm(index)

        for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(process_bar):
            try:
                pdb_path = os.path.join(self.raw_path, pocket_fn)
                sdf_path = os.path.join(self.raw_path, ligand_fn)

                if not os.path.exists(pdb_path):
                    logging.warning(f"PDB not found: {pdb_path}")
                    continue
                if not os.path.exists(sdf_path):
                    logging.warning(f"SDF not found: {pdb_path}")
                    continue

                if ligand_fn in preserve_name:
                    preserve = True
                else:
                    preserve = False

                tasks.append({
                    'pdb_path': pdb_path,
                    'sdf_path': sdf_path,
                    'pdb_entry': pocket_fn,
                    'sdf_entry': ligand_fn,
                    'force_preserve': preserve
                    })
            except:
                pass

        # [parse_protein_ligand_pairs(task)
        #     for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocessing paired protein ligand features')]

        data_list = joblib.Parallel(
            n_jobs = max(joblib.cpu_count() // 2, 1),
        )(
            joblib.delayed(parse_protein_ligand_pairs)(task)
            for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocessing paired protein ligand features')
        )

        db_conn = lmdb.open(
            self.processed_paired_path,
            map_size = self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        
        id = 0
        with db_conn.begin(write=True, buffers=True) as txn:
            for i, data in enumerate(tqdm(data_list, dynamic_ncols=True, desc='Writing to LMDB')):
                if data is None:
                    continue
                data['id'] = id
                txn.put(str(data['id']).encode('utf-8'), pickle.dumps(data))
                id += 1
        db_conn.close()

        print('Valid path number is {}'.format(len(tasks)))
        print('Valid protein-ligand pair number is {}'.format(id))


    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = data['entry']
            name2id[name] = i
        torch.save(name2id, self.name2id_path)
    
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    
    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        # assert data.protein_pos.size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        data['id'] = idx
        return data
    

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_paired_path,
            map_size=self.MAP_SIZE,   
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))