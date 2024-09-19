import os
from typing import Any, Dict

import numpy as np
import tqdm

from ..datasets.lmdb_dataset import LMDBDataset
from ..utils.utils import standardize_smiles

fpath = "/data02/gtguo/DEL/data/raw/001-sEH/total_compounds.txt"
output_path = "/data02/gtguo/DEL/data/lmdb/001_sEH.lmdb"


def line_to_dict_sample(line: str) -> Dict[str, Any]:
    # split line
    line = line.strip()
    output = line.split(",")
    smiles, readout, bb1_smiles, bb2_smiles, bb3_smiles, *_ = output
    # standardize smiles

    # smiles = standardize_smiles(smiles)
    # bb1_smiles = standardize_smiles(bb1_smiles)
    # bb2_smiles = standardize_smiles(bb2_smiles)
    # bb3_smiles = standardize_smiles(bb3_smiles)

    return {
        "smiles": smiles,
        "readout": {
            "sEH": {
                "target": np.array([readout], dtype=float),
                "control": np.array([0.0]),
            }
        },
        "bb1_smiles": bb1_smiles,
        "bb2_smiles": bb2_smiles,
        "bb3_smiles": bb3_smiles,
    }


with open(fpath, "r") as f:
    lines = f.readlines()
    dataset = [line_to_dict_sample(line) for line in tqdm.tqdm(lines[1:])]

# save dataset
dataset = LMDBDataset.static_from_others(dataset, *os.path.split(output_path))