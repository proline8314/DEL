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
from torch_geometric.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up tensorboard
tb_path = os.path.join(os.path.dirname(__file__), "runs")
if not os.path.exists(tb_path):
    os.makedirs(tb_path)
writer = SummaryWriter(tb_path)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up random seed
seed = 4
torch.manual_seed(seed)
np.random.seed(seed)

# Set up arguments
parser = argparse.ArgumentParser()
# general
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--log_interval", type=int, default=5)
parser.add_argument("--save_interval", type=int, default=5)
parser.add_argument("--save_path", type=str, default="/data02/gtguo/data/weights/")
parser.add_argument("--load_path", type=str, default=None)
# dataset
parser.add_argument("--target_name", type=str, default="ca9")
parser.add_argument("--fp_size", type=int, default=2048)
# model

# loss
parser.add_argument("--target_size", type=int, default=4)
parser.add_argument("--label_size", type=int, default=6)
parser.add_argument("--matrix_size", type=int, default=2)

args = parser.parse_args()

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
    del_train_dataset, batch_size=args.batch_size, shuffle=True
)
del_val_loader = DataLoader(del_val_dataset, batch_size=args.batch_size, shuffle=False)

# Model
encoder = DELRefEncoder(
    node_feat_dim=21,
    edge_feat_dim=2,
    node_embedding_size=64,
    edge_embedding_size=64,
    n_layers=3,
    gat_n_heads=4,
    gat_ffn_ratio=4,
    with_fp=True,
    fp_embedding_size=64,
    fp_ffn_size=128,
    fp_gated=True,
    fp_n_heads=4,
    fp_size=2048,
    fp_to_gat_feedback="add",
    gat_to_fp_pooling="mean",
).to(device)
decoder = DELRefDecoder(
    node_input_size=64,
    node_emb_size=64,
    fp_input_size=64,
    fp_emb_size=64,
    output_size=2,
    output_activation=torch.exp,
    with_fp=True,
).to(device)
model = DELRefNet(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = ZIPLoss(
    label_size=args.label_size,
    matrix_size=args.matrix_size,
    target_size=args.target_size,
).to(device)

# Load model
if args.load_path:
    model.load_state_dict(torch.load(args.load_path))

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
            logger.info(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
            writer.add_scalar(
                "train/loss", loss.item(), epoch * len(del_train_loader) + i
            )

    if epoch % args.save_interval == 0:
        torch.save(
            model.state_dict(), os.path.join(args.save_path, f"model_{epoch}.pt")
        )

    model.eval()
    with torch.no_grad():
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
            writer.add_scalar("val/loss", loss.item(), epoch * len(del_val_loader) + i)
            logger.info(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")

writer.close()

# Save model
torch.save(model.state_dict(), os.path.join(args.save_path, f"model_{args.epochs}.pt"))
