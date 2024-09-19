import argparse
import os

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from ..datasets.lmdb_dataset import LMDBDataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_fpath", type=str, required=True)
parser.add_argument("--output_fpath", type=str, required=True)
args = parser.parse_args()

def collate_fn(sample):
    """
    sample: {
        "bb_pyg_data": {"x": np.array, "edge_idx": np.array},
        "pyg_data": {"x": np.array, "edge_idx": np.array, "edge_attr": np.array},
        "readout": dict_containing_np_arrays,
    }
    """
    _sample = {}
    bb_pyg_data = Data(
        x=torch.LongTensor(sample["bb_pyg_data"]["x"]),
        edge_index=torch.LongTensor(sample["bb_pyg_data"]["edge_idx"]),
    )
    _sample["bb_pyg_data"] = bb_pyg_data
    pyg_data = Data(
        x=torch.tensor(sample["pyg_data"]["x"]),
        edge_index=torch.LongTensor(sample["pyg_data"]["edge_idx"]),
        edge_attr=torch.tensor(sample["pyg_data"]["edge_attr"]),
    )
    _sample["pyg_data"] = pyg_data
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
    dataset = LMDBDataset.static_from_raw(*os.path.split(args.dataset_fpath), *os.path.split(args.output_fpath), process_fn=collate_fn)
    print(dataset[0])
    print(len(dataset))