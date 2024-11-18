import os

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from ..datasets.lmdb_dataset import LMDBDataset
from ..utils.mol_feat_v2 import (SmilesFingerprint, get_edge_features,
                                 get_edge_index, get_node_features)
from ..utils.utils import get_mol_from_smiles

dataset_fpath = "/data03/gtguo/data/DEL/CA2/lmdb/002_CAIX_idx.lmdb"
output_fpath = "/data03/gtguo/data/DEL/CA2/lmdb/002_CAIX_feat.lmdb"

sfpg = SmilesFingerprint(radius=2, nBits=2048)

# linear backbone
"""
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
    [0, 1],
    [1, 0],
    [1, 2],
    [2, 1],
    [1, 3],
    [3, 1],
]


def feat_fn(sample: dict) -> dict:
    # pyg data
    try:
        mol = get_mol_from_smiles(sample["smiles"])
    except:
        mol = get_mol_from_smiles(sample["smiles"], sanitize=False)

    mol_x = get_node_features(mol)
    mol_edge_idx = get_edge_index(mol)
    mol_edge_attr = get_edge_features(mol)

    # fingerprint
    bb_fingerprints = [sfpg("")]

    for bb_idx in range(1, 4):
        bb_smiles = sample[f"bb{bb_idx}_smiles"]
        bb_fingerprint = sfpg(bb_smiles)
        bb_fingerprints.append(bb_fingerprint)

    bb_x = np.vstack(bb_fingerprints)

    _sample = {}
    _sample["bb_pyg_data"] = {
        "x": bb_x,
        "edge_index": np.array(bb_edge_idx).T,
    }
    _sample["pyg_data"] = {
        "x": mol_x,
        "edge_index": mol_edge_idx,
        "edge_attr": mol_edge_attr,
        "synthon_index" : sample["synthon_index"],
    }
    _sample["readout"] = sample["readout"]
    _sample["idx"] = sample["idx"]
    return _sample


def to_tensor(nested_dict: dict) -> dict:
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            to_tensor(v)
        else:
            nested_dict[k] = torch.tensor(v)
    return nested_dict


# LMDBDataset = LMDBDataset.update_process_fn(feat_fn)
if __name__ == "__main__":
    dataset = LMDBDataset.static_from_raw(
        *os.path.split(dataset_fpath),
        *os.path.split(output_fpath),
        nprocs=4,
        process_fn=feat_fn,
    )
    print(dataset[0])
    print(len(dataset))
