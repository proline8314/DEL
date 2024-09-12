import os

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .datasets.lmdb_dataset import LMDBDataset
from .models.ref_net_v2 import (BidirectionalPipe, GraphAttnEncoder,
                                PyGDataInputLayer, RefNetV2, RegressionHead)

path = "E:/Research/del/data/lmdb/002_CAIX_feat.lmdb"
new_path = "E:/Research/del/data/lmdb/002_CAIX.lmdb"
dataset = LMDBDataset.readonly_raw(*os.path.split(path), map_size=1024**3 * 16)


def dataset_fn(sample: dict) -> dict:
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
    print(dataset[0])
    print(dataset[0]["bb_pyg_data"].x.sum(axis=-1))
    print(len(dataset))


    # dataset = LMDBDataset.update_process_fn(dataset_fn).static_from_others(dataset, *os.path.split(new_path), map_size=1024**3 * 16)

    for data in tqdm(dataset):
        print(data)
        break

    raise NotImplementedError

    model = RefNetV2(
        PyGDataInputLayer(
            2048, 64, "Embedding-Linear", None, 64, "None", token_size=16
        ),
        PyGDataInputLayer(147, 64, "Linear", 5, 64, "Linear"),
        GraphAttnEncoder(64, 64),
        GraphAttnEncoder(64, 64),
        [BidirectionalPipe(64, 64) for _ in range(5)],
        RegressionHead(64, 64, 1, "sigmoid"),
        RegressionHead(64, 64, 2, None),
    )
    # number of parameters
    print(f"number of parameters: {sum(p.numel() for p in model.parameters())}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for data in tqdm(loader):
        print(data)
        print(data["bb_pyg_data"].batch)
        print(data["pyg_data"].batch)
        out = model(data["bb_pyg_data"], data["pyg_data"])
        print(out)
