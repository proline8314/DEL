import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import tensorboard as tb
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.chembl_dataset import FULL_NAME_DICT, ChemBLActivityDataset
from datasets.lmdb_dataset import LMDBDataset
from models.ref_net import DELRefDecoder, DELRefEncoder, DELRefNet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool as gap
from utils.mol_feat import process_to_pyg_data

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up arguments
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--name", type=str, default="ca9_full_-loss")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument(
        "--save_path", type=str, default="/data02/gtguo/DEL/data/weights/refnet/"
    )
    parser.add_argument(
        "--load_path", type=str, default="/data02/gtguo/DEL/data/weights/refnet/"
    )
    # dataset
    parser.add_argument("--target_name", type=str, default="ca9")
    parser.add_argument("--fp_size", type=int, default=2048)

    # dataloader
    parser.add_argument("--num_workers", type=int, default=8)
    # model encoder
    parser.add_argument("--enc_node_feat_dim", type=int, default=19)
    parser.add_argument("--enc_edge_feat_dim", type=int, default=2)
    parser.add_argument("--enc_node_embedding_size", type=int, default=64)
    parser.add_argument("--enc_edge_embedding_size", type=int, default=64)
    parser.add_argument("--enc_n_layers", type=int, default=3)
    parser.add_argument("--enc_gat_n_heads", type=int, default=4)
    parser.add_argument("--enc_gat_ffn_ratio", type=int, default=4)
    parser.add_argument("--enc_fp_embedding_size", type=int, default=32)
    parser.add_argument("--enc_fp_ffn_size", type=int, default=128)
    parser.add_argument("--enc_fp_gated", action="store_true")
    parser.add_argument("--enc_fp_n_heads", type=int, default=4)
    parser.add_argument("--enc_fp_size", type=int, default=256)
    parser.add_argument("--enc_fp_to_gat_feedback", type=str, default="add")
    parser.add_argument("--enc_gat_to_fp_pooling", type=str, default="mean")

    # model decoder
    parser.add_argument("--dec_node_input_size", type=int, default=64)
    parser.add_argument("--dec_node_emb_size", type=int, default=64)
    parser.add_argument("--dec_fp_input_size", type=int, default=32)
    parser.add_argument("--dec_fp_emb_size", type=int, default=64)
    parser.add_argument("--dec_output_size", type=int, default=2)

    # loss
    parser.add_argument("--target_size", type=int, default=4)
    parser.add_argument("--label_size", type=int, default=6)
    parser.add_argument("--matrix_size", type=int, default=2)

    # record
    parser.add_argument(
        "--record_path", type=str, default="/data02/gtguo/DEL/data/records/refnet/"
    )

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
    encoder = DELRefEncoder(
        node_feat_dim=args.enc_node_feat_dim,
        edge_feat_dim=args.enc_edge_feat_dim,
        node_embedding_size=args.enc_node_embedding_size,
        edge_embedding_size=args.enc_edge_embedding_size,
        n_layers=args.enc_n_layers,
        gat_n_heads=args.enc_gat_n_heads,
        gat_ffn_ratio=args.enc_gat_ffn_ratio,
        with_fp=False,
        fp_embedding_size=args.enc_fp_embedding_size,
        fp_ffn_size=args.enc_fp_ffn_size,
        fp_gated=args.enc_fp_gated,
        fp_n_heads=args.enc_fp_n_heads,
        fp_size=args.enc_fp_size,
        fp_to_gat_feedback=args.enc_fp_to_gat_feedback,
        gat_to_fp_pooling=args.enc_gat_to_fp_pooling,
    ).to(device)
    decoder = DELRefDecoder(
        node_input_size=args.dec_node_input_size,
        node_emb_size=args.dec_node_emb_size,
        fp_input_size=args.dec_fp_input_size,
        fp_emb_size=args.dec_fp_emb_size,
        output_size=args.dec_output_size,
        output_activation=torch.exp,
        with_fp=False,
        with_dist=False,
    ).to(device)
    model = DELRefNet(encoder, decoder).to(device)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Datasets
    def process(sample):
        smiles = sample["mol_structures"]["smiles"]
        pyg_data = process_to_pyg_data(smiles)
        activity = sample[FULL_NAME_DICT[args.target_name]]["activity"]
        return {"pyg_data": pyg_data, "activity": activity}

    chembl_dataset = ChemBLActivityDataset(
        FULL_NAME_DICT[args.target_name], update_target=False
    )
    active_dataset, _ = chembl_dataset.split_with_condition(
        lambda is_hit: is_hit == 1,
        (("Carbonic anhydrase IX", "is_hit"),),
    )
    active_dataset = LMDBDataset.update_process_fn(
        process_fn=process
    ).dynamic_from_others(active_dataset)
    logger.info(f"Dataset has {len(active_dataset)} samples")
    logger.info(f"Dataset data example: {active_dataset[0]}")

    # DataLoaders
    data_loader = DataLoader(
        active_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Load model
    if args.load_path:
        model.load_state_dict(
            torch.load(os.path.join(args.load_path, args.name, "model.pt")),
            strict=False,
        )

    # Test
    model.eval()
    output_data = []
    activity_data = []

    for batch in data_loader:
        data = batch["pyg_data"].to(device)
        activity = batch["activity"].to(device)
        output, _, _ = encoder(
            data.x, data.edge_index, data.edge_attr, None, data.batch, None
        )
        output = gap(decoder.n_dec(output), data.batch)
        output_data.append(output)
        activity_data.append(activity)

    output_data = torch.cat(output_data, dim=0).detach().cpu().numpy()
    activity_data = torch.cat(activity_data, dim=0).detach().cpu().numpy()

    # linear regression
    reg = LinearRegression().fit(output_data, activity_data)
    r2 = reg.score(output_data, activity_data)
    logger.info(f"R2: {r2}")
