import os
import pickle
from functools import lru_cache

import lmdb
import numpy as np
import rdkit
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.SaltRemover import SaltRemover
from tqdm import tqdm


class BuildingBlockFingerprintDataset(torch.utils.data.Dataset):
    def __init__(self,
                 target_name: str = "ca9",
                 radius: int = 2,
                 fpsize: int = 2048,
                 root="/data02/gtguo/DEL/data/dataset/acs.jcim.2c01608/raw",
                 fname="features.lmdb",
                 processed_root="/data02/gtguo/DEL/data/dataset/acs.jcim.2c01608/processed",
                 processed_fname="features_bbfp.lmdb"):
        self.target_name = target_name
        self.fpsize = fpsize
        self.raw_fpath = os.path.join(root, fname)
        self.processed_fpath = os.path.join(processed_root, processed_fname)
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=fpsize)
        self.salt_remover = SaltRemover()

        if not os.path.exists(self.processed_fpath):
            self._process()
        
        self.env = lmdb.open(self.processed_fpath, subdir=False,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False,
                            max_readers=1,
                            map_size=1099511627776)
        self.txn = self.env.begin(write=False)


    def __len__(self):
        if not hasattr(self, "_keys"):
            with lmdb.open(self.processed_fpath, subdir=False,
                           readonly=True,
                           lock=False,
                           readahead=False,
                           meminit=False,
                           max_readers=1,
                           map_size=1099511627776).begin() as txn:
                self._keys = list(txn.cursor().iternext(values=False))
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return pickle.loads(self.txn.get(f"{idx}".encode()))

    def _process(self):
        new_env = lmdb.open(self.processed_fpath, subdir=False,
                            readonly=False,
                            lock=False,
                            readahead=False,
                            meminit=False,
                            max_readers=1,
                            map_size=1099511627776)
        new_txn = new_env.begin(write=True)
        with lmdb.open(self.raw_fpath, subdir=False,
                       readonly=False,
                       lock=False,
                       readahead=False,
                       meminit=False,
                       max_readers=1,
                       map_size=1099511627776).begin() as txn:
            keys = list(txn.cursor().iternext(values=False))
            self._keys = keys
            for idx in tqdm(range(len(keys))):
                data_dict = pickle.loads(txn.get(f"{idx}".encode()))
                BB1_mol = self._mol_from_smiles(data_dict["BB1_smiles"])
                BB2_mol = self._mol_from_smiles(data_dict["BB2_smiles"])
                BB3_mol = self._mol_from_smiles(data_dict["BB3_smiles"])

                BB1_fp = self.mfpgen.GetFingerprint(BB1_mol)
                BB2_fp = self.mfpgen.GetFingerprint(BB2_mol)
                BB3_fp = self.mfpgen.GetFingerprint(BB3_mol)

                BB1_fp = np.array(BB1_fp, dtype=np.int64)
                BB2_fp = np.array(BB2_fp, dtype=np.int64)
                BB3_fp = np.array(BB3_fp, dtype=np.int64)

                BB1_fp = torch.tensor(BB1_fp, dtype=torch.float)
                BB2_fp = torch.tensor(BB2_fp, dtype=torch.float)
                BB3_fp = torch.tensor(BB3_fp, dtype=torch.float)

                target = self._get_target(data_dict)

                new_txn.put(str(idx).encode(),
                            pickle.dumps(
                                {
                                    "input":
                                    {
                                        "BB1_fp": BB1_fp,
                                        "BB2_fp": BB2_fp,
                                        "BB3_fp": BB3_fp
                                    },
                                    "target":
                                    {
                                        "target": target
                                    }

                                },
                                protocol=-1
                )
                )

        new_txn.commit()
        new_env.close()

    @lru_cache(maxsize=16)
    def _mol_from_smiles(self, smiles: str):
        if not isinstance(smiles, str):
            smiles = ""
        mol = Chem.MolFromSmiles(smiles)
        mol = self.salt_remover.StripMol(mol)
        return mol

    def _get_target(self, data_dict):
        label = np.concatenate(
            (data_dict[f"{self.target_name}_target"], data_dict[f"{self.target_name}_matrix"]))
        return torch.tensor(label, dtype=torch.float)

if __name__ == "__main__":
    dataset = BuildingBlockFingerprintDataset()
    print(len(dataset))
    print(dataset[0])
    print(dataset[0]["input"]["BB1_fp"].sum())