import os
import sys
from glob import glob

import numpy as np
import rdkit
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
from tqdm import tqdm

from .lmdb_dataset import LMDBDataset


class ChemDivDataset(LMDBDataset):
    def __init__(self, dataset_dir="/data03/gtguo/data/chemdiv/lmdb/chemdiv.lmdb"):
        self._dataset = ChemDivSMILESDataset()
        fdir, fname = os.path.split(dataset_dir)
        super(ChemDivDataset, self).__init__(
            source_dataset=self._dataset,
            processed_dir=fdir,
            processed_fname=fname,
            source="others",
        )
        

class ChemDivSMILESDataset(Dataset):
    def __init__(self, chemdiv_sdf_dir="/data03/gtguo/data/chemdiv/processed_sdf/"):
        super(ChemDivSMILESDataset, self).__init__()
        self.chemdiv_sdf_dir = chemdiv_sdf_dir
        self.sdf_files = glob(self.chemdiv_sdf_dir + "*.sdf")

        self._data = []

        for sdf_file in self.sdf_files:
            suppl = Chem.SDMolSupplier(sdf_file)
            for mol in tqdm(suppl):
                if mol is None:
                    raise ValueError("Molecule is None")
                sample = {}
                sample["smiles"] = Chem.MolToSmiles(mol)
                sample["source"] = os.path.basename(sdf_file).split(".")[0]
                self._data.append(sample)

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        return self._data[idx]
