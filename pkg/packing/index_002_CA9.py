import os
import warnings
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from ..datasets.lmdb_dataset import LMDBDataset
from ..utils.get_synthon_index import get_synthon_index
from ..utils.utils import get_mol_from_smiles, standardize_smiles

# close warnings
warnings.filterwarnings("ignore")

dataset_fpath = "/data03/gtguo/data/DEL/CA2/lmdb/002_CAIX.lmdb"
output_fpath = "/data03/gtguo/data/DEL/CA2/lmdb/002_CAIX_idx.lmdb"


def handle_sample(sample: dict) -> dict:
    # read smiles
    smiles = sample["smiles"]
    mol = get_mol_from_smiles(smiles)

    syn_mol_dict = {
        i: get_mol_from_smiles(sample[f"bb{i}_smiles"])
        for i in range(1, 4)
        if type(sample[f"bb{i}_smiles"]) != float
    }

    # get synthon index
    synthon_index = get_synthon_index(mol, syn_mol_dict)

    # update sample
    sample["synthon_index"] = synthon_index
    return sample


dataset = LMDBDataset.update_process_fn(handle_sample).static_from_raw(
    *os.path.split(dataset_fpath), *os.path.split(output_fpath)
)
print(dataset[0])
print(len(dataset))
