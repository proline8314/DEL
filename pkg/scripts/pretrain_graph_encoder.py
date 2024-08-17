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
from datasets.chembl_dataset import ChemBLMolSmilesDataset
from datasets.lmdb_dataset import LMDBDataset
from models.ref_net import DELRefEncoder, RegressionHead
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from utils.mol_feat import process_to_pyg_data

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up arguments
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--name", type=str, default="ca9_full")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument(
        "--save_path",
        type=str,
        default="/data02/gtguo/DEL/data/weights/refnet_encoder_pretrain/",
    )
    parser.add_argument("--load_path", type=str, default=None)
    # dataset

    # data split
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--valid_size", type=float, default=0.2)
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

    # model regression head
    parser.add_argument("--node_head_input_size", type=int, default=64)
    parser.add_argument("--node_head_hidden_size", type=int, default=64)
    parser.add_argument("--node_head_output_size", type=int, default=19)
    parser.add_argument("--edge_head_input_size", type=int, default=64)
    parser.add_argument("--edge_head_hidden_size", type=int, default=64)
    parser.add_argument("--edge_head_output_size", type=int, default=2)

    # loss

    # record
    parser.add_argument(
        "--record_path",
        type=str,
        default="/data02/gtguo/DEL/data/records/refnet_encoder_pretrain/",
    )

    args = parser.parse_args()

    # print arguments to log
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # Set up tensorboard
    tb_path = os.path.join(
        os.path.dirname(__file__), os.path.join(args.record_path, args.name)
    )
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    writer = SummaryWriter(tb_path)

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
        fp_embedding_size=None,
        fp_ffn_size=None,
        fp_gated=None,
        fp_n_heads=None,
        fp_size=None,
        fp_to_gat_feedback=None,
        gat_to_fp_pooling=None,
    ).to(device)
    node_head = RegressionHead(
        input_size=args.node_head_input_size,
        hidden_size=args.node_head_hidden_size,
        output_size=args.node_head_output_size,
    ).to(device)
    edge_head = RegressionHead(
        input_size=args.edge_head_input_size,
        hidden_size=args.edge_head_hidden_size,
        output_size=args.edge_head_output_size,
    ).to(device)

    logger.info(
        f"Encoder has {sum(p.numel() for p in encoder.parameters())} parameters"
    )
    logger.info(
        f"Node Head has {sum(p.numel() for p in node_head.parameters())} parameters"
    )
    logger.info(
        f"Edge Head has {sum(p.numel() for p in edge_head.parameters())} parameters"
    )

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        list(encoder.parameters())
        + list(node_head.parameters())
        + list(edge_head.parameters()),
        lr=args.lr,
    )

    # Datasets
    def process(sample):
        return process_to_pyg_data(sample["mol_structures"]["smiles"])

    chembl_dataset = LMDBDataset.update_process_fn(process).static_from_others(
        dataset=ChemBLMolSmilesDataset(),
        processed_dir="/data02/gtguo/DEL/data/dataset/chembl",
        processed_fname="chembl_mol_data.lmdb",
    )
    # DataLoaders
    train_data, val_data = train_test_split(
        chembl_dataset, train_size=args.train_size, test_size=args.valid_size, random_state=seed, shuffle=True
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


    # Load model
    if args.load_path:
        encoder.load_state_dict(torch.load(args.load_path))

    if not os.path.exists(os.path.join(args.save_path, args.name)):
        os.makedirs(os.path.join(args.save_path, args.name))

    # Train

    for epoch in range(args.epochs):
        encoder.train()
        node_head.train()
        edge_head.train()
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            node_embedding, edge_embedding = encoder(
                data.x, data.edge_index, data.edge_attr
            )
            node_output = node_head(node_embedding)
            edge_output = edge_head(edge_embedding)
            loss = criterion(
                torch.cat((node_output, edge_output), dim=1),
                torch.cat(
                    (
                        data.y_node.view(data.batch_size, args.enc_node_feat_dim),
                        data.y_edge.view(data.batch_size, args.enc_edge_feat_dim),
                    ),
                    dim=1,
                ),
            )
            loss.backward()
            optimizer.step()
            if i % args.log_interval == 0:
                logger.info(f"Epoch {epoch}, Iteration {i}, Train Loss: {loss.item()}")
                writer.add_scalar(
                    "train/loss", loss.item(), epoch * len(train_loader) + i
                )

        if epoch % args.save_interval == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.save_path, args.name, f"model_{epoch}.pt"),
            )

        model.eval()
        with torch.no_grad():
            losses = []
            for i, data in enumerate(del_val_loader):
                data = data.to(device)
                output = model(
                    data.x,
                    data.edge_index,
                    data.edge_attr,
                    data.node_bbidx,
                    data.batch,
                    data.bbfp.view(data.batch_size, -1, args.fp_size),
                    data.node_dist,
                )
                loss = criterion(
                    output,
                    torch.cat(
                        (
                            data.y_target.view(data.batch_size, args.target_size),
                            data.y_matrix.view(data.batch_size, args.matrix_size),
                        ),
                        dim=1,
                    ),
                )
                losses.append(loss.item())
            val_loss = np.mean(losses)
            logger.info(f"Epoch {epoch}, Validation Loss: {val_loss}")
            writer.add_scalar("val/loss", val_loss, epoch)

    # loss parameters
    logger.info(
        f"Loss parameters: {criterion.total_count, criterion.target_inflat_prob, criterion.matrix_inflat_prob}"
    )

    writer.close()

    # Save model
    torch.save(model.state_dict(), os.path.join(args.save_path, args.name, "model.pt"))
