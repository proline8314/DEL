import logging
import os
import sqlite3
from functools import lru_cache

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.SaltRemover import SaltRemover
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from .lmdb_dataset import LMDBDataset

FDIR = "/data03/gtguo/data/chembl/lmdb"
FNAME = "chembl_activity.lmdb"
DATABASE_PATH = "/data02/gtguo/data/chembl_33/chembl_33_sqlite/chembl_33.db"

CHEMBL_ID_NAME_DICT = {
    "CHEMBL612545": "Horseradish peroxidase"
}
FULL_NAME_DICT = {
    "ca9": "Carbonic anhydrase IX",
    "hrp": "Horseradish peroxidase",
    "ca2": "Carbonic anhydrase II",
    "ca12": "Carbonic anhydrase XII",
    "sEH": "Epoxide hydratase",
    "seh": "Epoxide hydratase",
}

# ! TODO: readonly mode (block _process by alternative __init__ method?)

class ChemBLActivityDataset(LMDBDataset):
    def __init__(self, target_name: str = "Carbonic anhydrase IX", update_target: bool = True, *, target_chembl_id: str = None):
        if target_name is not None:
            self.target_name = target_name
            statement = f""" AND td.pref_name = '{self.target_name}';"""
        elif target_chembl_id is not None and target_chembl_id in CHEMBL_ID_NAME_DICT:
            self.target_name = CHEMBL_ID_NAME_DICT[target_chembl_id]
            statement = f""" AND td.chembl_id = '{target_chembl_id}';"""
        else:
            raise ValueError("Unrecogized target_name or target_chembl_id")
        
        if update_target:
            self.con = sqlite3.connect(DATABASE_PATH)
            self.std_type = ("Kd", "Ki", "IC50", "GI50", "EC50")
            self.sql = f"""SELECT md.chembl_id,
                act.standard_type,
                act.standard_value
                FROM target_dictionary td
                JOIN assays a ON td.tid = a.tid
                JOIN activities act ON a.assay_id = act.assay_id
                JOIN molecule_dictionary md ON md.molregno = act.molregno
                JOIN compound_structures cs ON md.molregno = cs.molregno
                WHERE act.standard_relation     = '='
                AND act.standard_type         IN {self.std_type}
                AND act.standard_units        = 'nM';"""

            self.cursor = self.con.cursor().execute(self.sql[:-1] + statement)
            self.activity = [
                {
                    "molecule_chembl_id": row[0],
                    "standard_type": row[1],
                    "standard_value": row[2],
                    "activity": 9.0 - np.log10(float(row[2])),
                }
                for row in tqdm(self.cursor, desc="Loading ChemBL Activity") if row[2] is not None
            ]
            self.activity_mols_id = set(
                [act["molecule_chembl_id"] for act in self.activity]
            )

        if not os.path.exists(os.path.join(FDIR, FNAME)):
            self._dataset = ChemBLMolSmilesDataset()
            super(ChemBLActivityDataset, self).__init__(
                source_dataset=self._dataset,
                processed_dir=FDIR,
                processed_fname=FNAME,
                source="others",
            )
        else:
            # override
            super(ChemBLActivityDataset, self).__init__(
                raw_dir=FDIR,
                raw_fname=FNAME,
                processed_dir=FDIR,
                processed_fname=FNAME,
                source="raw",
                forced_process=update_target,
            )

    def process(self, sample):
        if not hasattr(self, "_process_fn"):
            if self._check_has_key(self.target_name, sample):
                print(f"Data for target \"{self.target_name}\" already exists")
                self._process_fn = lambda x: x
            else:
                print(f"Processing data for target \"{self.target_name}\"")
                assert hasattr(self, "activity")
                self._process_fn = self.process_new_target
        return self._process_fn(sample)

    def process_new_target(self, sample):
        sample[self.target_name] = {
            "is_hit": 0,
            "activity": 0.,
        }
        mol_chembl_id = sample["molecule_chembl_id"]
        mol_is_hit = mol_chembl_id in self.activity_mols_id

        if mol_is_hit:
            sample[self.target_name]["is_hit"] = 1
            activity = [
                act
                for act in self.activity
                if act["molecule_chembl_id"] == mol_chembl_id
            ]
            activity.sort(key=lambda x: self.std_type.index(x["standard_type"]))
            sample[self.target_name]["activity"] = activity[0]["activity"]
        return sample


class ChemBLMolSmilesDataset(Dataset):
    def __init__(self):
        self.con = sqlite3.connect(DATABASE_PATH)
        self.cursor = self.con.cursor().execute(
            "SELECT md.chembl_id, cs.canonical_smiles FROM molecule_dictionary md JOIN compound_structures cs ON md.molregno = cs.molregno"
        )
        self._data = {
            i: {"molecule_chembl_id": row[0], "mol_structures": {"smiles": row[1]}}
            for i, row in enumerate(self.cursor)
        }

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
