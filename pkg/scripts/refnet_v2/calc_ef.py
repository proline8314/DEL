import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ...datasets.lmdb_dataset import LMDBDataset
from ...models.ref_net_v2 import (BidirectionalPipe, GraphAttnEncoder,
                                  PyGDataInputLayer, RefNetV2, RegressionHead)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up arguments
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--name", type=str, default="ca9_czip")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=3072)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--weight_path", type=str, default="")
    # dataset
    parser.add_argument("--positive_dataset_fpath", type=str, default="")
    parser.add_argument("--negative_dataset_fpath", type=str, default="")
    parser.add_argument("--map_size", type=int, default=1024**3 * 16)

    # dataloader
    parser.add_argument("--num_workers", type=int, default=8)

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

    print("Reading dataset")
    # * pass feat_fn to LMDBDataset
    positive_dataset = LMDBDataset.readonly_raw(
        args.positive_dataset_fpath, map_size=args.map_size
    )
    negative_dataset = LMDBDataset.readonly_raw(
        args.negative_dataset_fpath, map_size=args.map_size
    )

    if not os.path.exists(os.path.join(args.save_path, args.name)):
        os.makedirs(os.path.join(args.save_path, args.name))

    # Train

    def move_data_to_device(data):
        for k, v in data.items():
            if isinstance(v, dict):
                move_data_to_device(v)
            else:
                data[k] = v.to(device)
        return data

    def inference_forward(net: RefNetV2, data):
        molecule_node_vec, molecule_edge_idx, molecule_edge_vec = (
            net.molecule_feat_input(data)
        )
        for i in range(net.n_layers):
            molecule_node_vec, molecule_edge_vec = net.molecule_encoder.layers[i](
                molecule_node_vec, molecule_edge_idx, molecule_edge_vec
            )
            molecule_node_vec[molecule_node_vec.isnan()] = 0.0

        affinity_pred = net.get_affinity(molecule_node_vec, data.batch)
        return affinity_pred
    
    def enrichment_factor()
