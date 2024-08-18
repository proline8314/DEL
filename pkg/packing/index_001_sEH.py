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

dataset_fpath = "E:/Research/del/data/lmdb/001_sEH.lmdb"


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


if __name__ == "__main__":
    nproc = 8

    dataset = LMDBDataset.readonly_raw(*os.path.split(dataset_fpath))
    updated_dataset = {}
    with Pool(nproc) as pool:
        for sample in tqdm(
            pool.imap_unordered(handle_sample, dataset, chunksize=100),
            total=len(dataset),
        ):
            updated_dataset[sample["smiles"]] = sample

    # save updated dataset
    def corresponding_fn(sample):
        return updated_dataset[sample["smiles"]]

    dataset = LMDBDataset.update_process_fn(corresponding_fn).override_raw(
        *os.path.split(dataset_fpath)
    )
    print(dataset[0])
