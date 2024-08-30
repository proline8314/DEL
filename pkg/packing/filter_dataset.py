import argparse
import os

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from ..datasets.lmdb_dataset import LMDBDataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_fpath", type=str, required=True)
parser.add_argument("--output_fpath", type=str, required=True)
parser.add_argument("--nproc", type=int, default=4)
args = parser.parse_args()

def filter_fn(sample: dict) -> dict:
    _sample = {}

    bb_pyg_data = sample["bb_pyg_data"]
    bb_pyg_data.x = torch.LongTensor(bb_pyg_data.x)
    _sample["bb_pyg_data"] = bb_pyg_data

    _pyg_data = sample["pyg_data"]
    _pyg_data["synthon_index"] = torch.LongTensor(sample["synthon_index"])
    _sample["pyg_data"] = _pyg_data

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
    dataset = LMDBDataset.static_from_raw(*os.path.split(args.dataset_fpath), *os.path.split(args.output_fpath), nprocs=args.nproc, process_fn=filter_fn)
    print(dataset[0])