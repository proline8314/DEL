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
from datasets.graph_dataset import GraphDataset
from losses.zip_loss import ZIPLoss
from models.ref_net import DELRefDecoder, DELRefEncoder, DELRefNet
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

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
        "--save_path", type=str, default="/data02/gtguo/DEL/data/weights/refnet/"
    )
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--update_loss", action="store_true")
    # dataset
    parser.add_argument("--target_name", type=str, default="ca9")
    parser.add_argument("--fp_size", type=int, default=2048)

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
    parser.add_argument("--enc_with_fp", action="store_true")
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
    parser.add_argument("--dec_with_fp", action="store_true")
    parser.add_argument("--dec_with_dist", action="store_true")

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
        with_fp=args.enc_with_fp,
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
        with_fp=args.dec_with_fp,
        with_dist=args.dec_with_dist,
    ).to(device)
    model = DELRefNet(encoder, decoder).to(device)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    criterion = ZIPLoss(
        label_size=args.label_size,
        matrix_size=args.matrix_size,
        target_size=args.target_size,
    ).to(device)

    if args.update_loss:
        optimizer = optim.Adam(
            list(model.parameters()) + list(criterion.parameters()), lr=args.lr
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Datasets
    del_dataset = GraphDataset(
        forced_reload=False, target_name=args.target_name, fpsize=args.fp_size
    )
    chembl_dataset = ChemBLActivityDataset(
        FULL_NAME_DICT[args.target_name], update_target=False
    )
    del_train_dataset, del_val_dataset = train_test_split(
        del_dataset, test_size=0.2, random_state=seed
    )

    # DataLoaders
    del_train_loader = DataLoader(
        del_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    del_val_loader = DataLoader(del_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    # Load model
    if args.load_path:
        model.load_state_dict(torch.load(args.load_path))

    if not os.path.exists(os.path.join(args.save_path, args.name)):
        os.makedirs(os.path.join(args.save_path, args.name))

    # Train

    for epoch in range(args.epochs):
        model.train()
        for i, data in enumerate(del_train_loader):
            data = data.to(device)
            optimizer.zero_grad()
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
            loss.backward()
            optimizer.step()
            if i % args.log_interval == 0:
                logger.info(f"Epoch {epoch}, Iteration {i}, Train Loss: {loss.item()}")
                writer.add_scalar(
                    "train/loss", loss.item(), epoch * len(del_train_loader) + i
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
    logger.info(f"Loss parameters: {criterion.total_count, criterion.target_inflat_prob, criterion.matrix_inflat_prob}")

    writer.close()

    # Save model
    torch.save(model.state_dict(), os.path.join(args.save_path, args.name, "model.pt"))
