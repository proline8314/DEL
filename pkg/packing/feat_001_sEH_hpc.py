import os
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from ..datasets.lmdb_dataset import LMDBDataset
from ..utils.mol_feat_v2 import SmilesFingerprint, process_to_pyg_data

dataset_fpath = "E:/Research/del/data/lmdb/001_sEH.lmdb"
output_fpath = "E:/Research/del/data/lmdb/001_sEH_feat.lmdb"
nproc = 4

sfpg = SmilesFingerprint(radius=2, nBits=2048)

# linear backbone

bb_edge_idx = [
    [0, 1],
    [1, 0],
    [1, 2],
    [2, 1],
    [2, 3],
    [3, 2],
]
"""
# branched backbone
bb_edge_idx = [
    [0, 2],
    [2, 0],
    [2, 1],
    [1, 2],
    [2, 3],
    [3, 2],
]
"""


def feat_fn(sample: dict) -> dict:
    # pyg data
    mol_data = process_to_pyg_data(sample["smiles"])
    sample["pyg_data"] = mol_data

    # fingerprint
    bb_fingerprints = [sfpg("")]

    for bb_idx in range(1, 4):
        bb_smiles = sample[f"bb{bb_idx}_smiles"]
        bb_fingerprint = sfpg(bb_smiles)
        bb_fingerprints.append(bb_fingerprint)

    bb_x = np.vstack(bb_fingerprints)
    bb_data = Data(x=bb_x, edge_index=torch.tensor(bb_edge_idx, dtype=torch.long).T)
    sample["bb_pyg_data"] = bb_data

    _sample = {}
    _sample["bb_pyg_data"] = bb_data
    _sample["pyg_data"] = mol_data
    _sample["readout"] = to_tensor(sample["readout"])

    return _sample


def to_tensor(nested_dict: dict) -> dict:
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            to_tensor(v)
        else:
            nested_dict[k] = torch.tensor(v)
    return nested_dict


if __name__ == "__main__":
    dataset = LMDBDataset.static_from_raw(
        *os.path.split(dataset_fpath), *os.path.split(output_fpath), nprocs=nproc, process_fn=feat_fn
    )
    print(len(dataset))
    print(dataset[0])
