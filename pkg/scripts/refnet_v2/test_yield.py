import argparse
import logging
import os

import numpy as np
import tensorboard as tb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ...datasets.lmdb_dataset import LMDBDataset
from ...losses.zip_loss import CorrectedZIPLoss, ZIPLoss
from ...models.ref_net_v2 import (BidirectionalPipe, GraphAttnEncoder,
                                  PyGDataInputLayer, RefNetV2, RegressionHead)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up arguments
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--load_path", type=str, default="/data03/gtguo/del/refnet_weight/ca9_czip_correct_branch")
    # dataset
    parser.add_argument(
        "--dataset_fpath",
        type=str,
        default="/data03/gtguo/data/DEL/CA2/lmdb/002_CAIX_feat.lmdb",
    )
    parser.add_argument("--target_name", type=str, default="CA9")

    # dataloader
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--collate_dataset", action="store_true")

    # model
    parser.add_argument("--synthon_node_feat_dim", type=int, default=2048)
    parser.add_argument("--synthon_node_emb_dim", type=int, default=64)
    parser.add_argument(
        "--synthon_node_emb_method", type=str, default="Embedding-Linear"
    )
    parser.add_argument("--synthon_token_size", type=int, default=16)
    parser.add_argument("--synthon_edge_emb_dim", type=int, default=64)
    parser.add_argument("--mol_node_feat_dim", type=int, default=147)
    parser.add_argument("--mol_node_emb_dim", type=int, default=64)
    parser.add_argument("--mol_node_emb_method", type=str, default="Linear")
    parser.add_argument("--mol_edge_feat_dim", type=int, default=5)
    parser.add_argument("--mol_edge_emb_dim", type=int, default=64)
    parser.add_argument("--mol_edge_emb_method", type=str, default="Linear")
    parser.add_argument("--encoder_layers", type=int, default=5)

    # loss
    parser.add_argument("--target_size", type=int, default=4)
    parser.add_argument("--label_size", type=int, default=6)
    parser.add_argument("--matrix_size", type=int, default=2)

    args = parser.parse_args()

    # print arguments to log
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Set up random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Model
    synthon_handler = PyGDataInputLayer(
        args.synthon_node_feat_dim,
        args.synthon_node_emb_dim,
        args.synthon_node_emb_method,
        None,
        args.synthon_edge_emb_dim,
        "None",
        token_size=args.synthon_token_size,
    ).to(device)
    mol_handler = PyGDataInputLayer(
        args.mol_node_feat_dim,
        args.mol_node_emb_dim,
        args.mol_node_emb_method,
        args.mol_edge_feat_dim,
        args.mol_edge_emb_dim,
        args.mol_edge_emb_method,
    ).to(device)
    synthon_encoder = GraphAttnEncoder(
        args.synthon_node_emb_dim,
        args.synthon_edge_emb_dim,
        num_layers=args.encoder_layers,
    ).to(device)
    mol_encoder = GraphAttnEncoder(
        args.mol_node_emb_dim, args.mol_edge_emb_dim, num_layers=args.encoder_layers
    ).to(device)
    pipes = [
        BidirectionalPipe(args.synthon_node_emb_dim, args.mol_node_emb_dim).to(device)
        for _ in range(args.encoder_layers)
    ]
    yield_head = RegressionHead(
        args.synthon_edge_emb_dim, args.mol_edge_emb_dim, 1, "sigmoid"
    ).to(device)
    affinity_head = RegressionHead(
        args.mol_node_emb_dim, args.mol_node_emb_dim, 2, None
    ).to(device)

    model = RefNetV2(
        synthon_feat_input=synthon_handler,
        molecule_feat_input=mol_handler,
        synthon_encoder=synthon_encoder,
        molecule_encoder=mol_encoder,
        bidirectional_pipes=pipes,
        reaction_yield_head=yield_head,
        affinity_head=affinity_head,
    ).to(device)

    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Datasets
    class CollateDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return self.collate_fn(self.dataset[idx])
        
        def collate_fn(self, sample):
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
                edge_index=torch.LongTensor(sample["bb_pyg_data"]["edge_index"]),
            )
            _sample["bb_pyg_data"] = bb_pyg_data
            pyg_data = Data(
                x=torch.FloatTensor(sample["pyg_data"]["x"]),
                edge_index=torch.LongTensor(sample["pyg_data"]["edge_index"]),
                edge_attr=torch.FloatTensor(sample["pyg_data"]["edge_attr"]),
                synthon_index=torch.LongTensor(sample["pyg_data"]["synthon_index"]),
            )
            _sample["pyg_data"] = pyg_data
            _sample["readout"] = self.to_tensor(sample["readout"])
            _sample["idx"] = self.to_tensor(sample["idx"])

            return _sample

        def to_tensor(self, nested_dict: dict) -> dict:
            for k, v in nested_dict.items():
                if isinstance(v, dict):
                    self.to_tensor(v)
                else:
                    nested_dict[k] = torch.tensor(v)
            return nested_dict
        
    print("Reading dataset")
    del_dataset = LMDBDataset.readonly_raw(
        *os.path.split(args.dataset_fpath)
    )

    print("Splitting dataset")
    # idxs = np.arange(len(del_dataset))
    # train_idxs, val_idxs = train_test_split(
    #     idxs, train_size=args.train_size, test_size=args.valid_size
    # )
    # train_idxs = train_idxs.tolist()


    if args.collate_dataset:
        del_dataset = CollateDataset(del_dataset)

    print("Creating dataloaders")
    # DataLoaders
    del_test_loader = DataLoader(
        del_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Load model

    model.load_state_dict(torch.load(os.path.join(args.load_path, "model.pt")))

    # Infer

    # define yield forward function

    def yield_forward(model: RefNetV2, synthon_data, molecule_data):

        # * get the node and edge features from the input data
        synthon_node_vec, synthon_edge_idx, synthon_edge_vec = model.synthon_feat_input(
            synthon_data
        )
        molecule_node_vec, molecule_edge_idx, molecule_edge_vec = (
            model.molecule_feat_input(molecule_data)
        )

        # * encode the synthon and molecule graphs
        for i in range(model.n_layers):
            synthon_node_vec, synthon_edge_vec = model.synthon_encoder.layers[i](
                synthon_node_vec, synthon_edge_idx, synthon_edge_vec
            )
            molecule_node_vec, molecule_edge_vec = model.molecule_encoder.layers[i](
                molecule_node_vec, molecule_edge_idx, molecule_edge_vec
            )
            synthon_node_update = model.bidirectional_pipe[i].child2parent(
                synthon_node_vec,
                molecule_node_vec,
                synthon_data.batch,
                molecule_data.batch,
                molecule_data.synthon_index,
                molecule_data.ptr
            )
            molecule_node_update = model.bidirectional_pipe[i].parent2child(
                synthon_node_vec,
                molecule_node_vec,
                synthon_data.batch,
                molecule_data.batch,
                molecule_data.synthon_index,
                molecule_data.ptr
            )
            synthon_node_vec = synthon_node_vec + synthon_node_update
            molecule_node_vec = molecule_node_vec + molecule_node_update

            synthon_node_vec[synthon_node_vec.isnan()] = 0.0
            molecule_node_vec[molecule_node_vec.isnan()] = 0.0

        # * get the reaction yield and affinity predictions
        btz, _ = synthon_edge_vec.shape
        synthon_edge_batch = model._get_edge_batch(synthon_edge_idx, synthon_data.batch)

        synthon_edge_vec = synthon_edge_vec.view(btz // 2, 2, -1)[:, 0, :]
        synthon_edge_batch = synthon_edge_batch[::2]

        # * calculate the reaction yield
        yield_pred = model.reaction_yield_head(synthon_edge_vec)
        return yield_pred

    def move_data_to_device(data):
        for k, v in data.items():
            if isinstance(v, dict):
                move_data_to_device(v)
            else:
                data[k] = v.to(device)
        return data

    model.eval()
    model.to(device)

    output_yield = []
    output_scaffold_idx = []
    output_bb1_idx = []
    output_bb2_idx = []

    num_bb_edges = 3

    with torch.no_grad():
        for i, data in tqdm(enumerate(del_test_loader)):
            data = move_data_to_device(data)
            yield_pred = yield_forward(model, data["bb_pyg_data"], data["pyg_data"])
            output_yield.append(yield_pred.reshape(-1, num_bb_edges).cpu().numpy())
            output_scaffold_idx.append(data["idx"]["scaffold"].cpu().numpy())
            output_bb1_idx.append(data["idx"]["bb1"].cpu().numpy())
            output_bb2_idx.append(data["idx"]["bb2"].cpu().numpy())

    output_yield = np.concatenate(output_yield, axis=0)
    output_scaffold_idx = np.concatenate(output_scaffold_idx)
    output_bb1_idx = np.concatenate(output_bb1_idx)
    output_bb2_idx = np.concatenate(output_bb2_idx)

    if not os.path.exists("/data03/gtguo/del/refnetv2_yield_pred/czip_cb"):
        os.makedirs("/data03/gtguo/del/refnetv2_yield_pred/czip_cb")

    np.savez_compressed(
        "/data03/gtguo/del/refnetv2_yield_pred/czip_cb/test_yield_output_per_edges.npz",
        yield_pred=output_yield,
        scaffold_idx=output_scaffold_idx,
        bb1_idx=output_bb1_idx,
        bb2_idx=output_bb2_idx,
    )
            