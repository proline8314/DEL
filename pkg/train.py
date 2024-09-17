import argparse
import logging
import os

import numpy as np
import tensorboard as tb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .datasets.lmdb_dataset import LMDBDataset
from .losses.zip_loss import CorrectedZIPLoss, ZIPLoss
from .models.ref_net_v2 import (BidirectionalPipe, GraphAttnEncoder,
                                PyGDataInputLayer, RefNetV2, RegressionHead)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up arguments
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--name", type=str, default="ca9_zip")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=3072)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="E:\Research\del\data\weights")
    parser.add_argument("--update_loss", action="store_true")
    # dataset
    parser.add_argument(
        "--dataset_fpath",
        type=str,
        default="E:/Research/del/data/lmdb/002_CAIX_feat.lmdb",
    )
    parser.add_argument("--target_name", type=str, default="CA9")
    parser.add_argument("--map_size", type=int, default=1024**3 * 16)

    # data split
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--valid_size", type=float, default=0.2)

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
    parser.add_argument("--loss_sigma_correction", action="store_true")

    # record
    parser.add_argument(
        "--record_path", type=str, default="E:/Research/del/data/records/refnet"
    )

    # scheduler
    parser.add_argument("--lr_schedule", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--decay", type=float, default=0.1)
    parser.add_argument("--warmup_lr", type=float, default=2e-3)

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

    # Loss
    if args.loss_sigma_correction:
        criterion = CorrectedZIPLoss(
            label_size=args.label_size,
            matrix_size=args.matrix_size,
            target_size=args.target_size,
        ).to(device)
    else:
        criterion = ZIPLoss(
            label_size=args.label_size,
            matrix_size=args.matrix_size,
            target_size=args.target_size,
        ).to(device)

    if args.update_loss:
        optimizer = optim.Adam(
            list(model.parameters()) + list(criterion.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    # lr scheduler
    def lambda_lr(epoch, total_epoch, warmup_lr, min_lr, warmup_ratio=0.1, decay=0.1):
        assert warmup_ratio < 1
        assert decay < 1
        if epoch / total_epoch < warmup_ratio:
            return min_lr + (warmup_lr - min_lr) * epoch / (total_epoch * warmup_ratio)
        else:
            return min_lr + (warmup_lr - min_lr) * (1 - decay) ** (
                epoch - total_epoch * warmup_ratio
            )

    scheduler = None
    if args.lr_schedule:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: lambda_lr(
                epoch,
                args.epochs,
                args.warmup_lr / args.lr,
                args.lr / args.lr,
                args.warmup_ratio,
                args.decay,
            ),
            last_epoch=-1,
        )

    # Datasets
    print("Reading dataset")
    del_dataset = LMDBDataset.readonly_raw(
        *os.path.split(args.dataset_fpath), map_size=args.map_size
    )
    print("Splitting dataset")
    idxs = np.arange(len(del_dataset))
    train_idxs, val_idxs = train_test_split(
        idxs, train_size=args.train_size, test_size=args.valid_size
    )
    train_idxs = train_idxs.tolist()
    print("Creating dataloaders")
    del_train_dataset, del_val_dataset = del_dataset.split_with_idx(train_idxs)

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
            x=torch.LongTensor(sample["pyg_data"]["x"]),
            edge_index=torch.LongTensor(sample["pyg_data"]["edge_idx"]),
            edge_attr=torch.LongTensor(sample["pyg_data"]["edge_attr"]),
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

    # DataLoaders
    del_train_loader = DataLoader(
        del_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn if args.collate_dataset else None,
    )
    del_val_loader = DataLoader(
        del_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn if args.collate_dataset else None,
    )

    # Load model

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

    for epoch in range(args.epochs):
        model.train()
        for i, data in enumerate(del_train_loader):
            data = move_data_to_device(data)
            optimizer.zero_grad()
            output = model(data["bb_pyg_data"], data["pyg_data"])
            y = torch.cat(
                (
                    data["readout"][args.target_name]["target"],
                    data["readout"][args.target_name]["control"],
                ),
                dim=1,
            )
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            if i % args.log_interval == 0:
                logger.info(
                    f"Epoch {epoch}, Iteration {i}, Train Loss: {loss.detach().cpu().item()}, LR: {optimizer.param_groups[0]['lr']}"
                )
                writer.add_scalar(
                    "train/loss",
                    loss.detach().cpu().item(),
                    epoch * len(del_train_loader) + i,
                )
                if args.loss_sigma_correction:
                    _, tgt_nll, mat_nll, tgt_mse, mat_mse = criterion(
                        output, y, return_loss=True
                    )
                    writer.add_scalar(
                        "train/tgt_nll",
                        tgt_nll.detach().cpu().item(),
                        epoch * len(del_train_loader) + i,
                    )
                    writer.add_scalar(
                        "train/mat_nll",
                        mat_nll.detach().cpu().item(),
                        epoch * len(del_train_loader) + i,
                    )
                    writer.add_scalar(
                        "train/tgt_mse",
                        tgt_mse.detach().cpu().item(),
                        epoch * len(del_train_loader) + i,
                    )
                    writer.add_scalar(
                        "train/mat_mse",
                        mat_mse.detach().cpu().item(),
                        epoch * len(del_train_loader) + i,
                    )
                writer.add_scalar(
                    "train/lr",
                    optimizer.param_groups[0]["lr"],
                    epoch * len(del_train_loader) + i,
                )
        if args.lr_schedule:
            scheduler.step()

        if epoch % args.save_interval == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.save_path, args.name, f"model_{epoch}.pt"),
            )

        model.eval()
        with torch.no_grad():
            losses = []
            if args.loss_sigma_correction:
                tgt_nlls = []
                mat_nlls = []
                tgt_mses = []
                mat_mses = []
            for i, data in enumerate(del_val_loader):
                data = move_data_to_device(data)
                output = model(data["bb_pyg_data"], data["pyg_data"])
                if args.loss_sigma_correction:
                    loss, tgt_nll, mat_nll, tgt_mse, mat_mse = criterion(
                        output,
                        torch.cat(
                            (
                                data["readout"][args.target_name]["target"],
                                data["readout"][args.target_name]["control"],
                            ),
                            dim=1,
                        ),
                        return_loss=True,
                    )
                    losses.append(loss.detach().cpu().item())
                    tgt_nlls.append(tgt_nll.detach().cpu().item())
                    mat_nlls.append(mat_nll.detach().cpu().item())
                    tgt_mses.append(tgt_mse.detach().cpu().item())
                    mat_mses.append(mat_mse.detach().cpu().item())
                else:
                    loss = criterion(
                        output,
                        torch.cat(
                            (
                                data["readout"][args.target_name]["target"],
                                data["readout"][args.target_name]["control"],
                            ),
                            dim=1,
                        ),
                    )
                    losses.append(loss.detach().cpu().item())
            val_loss = np.mean(losses)
            logger.info(f"Epoch {epoch}, Validation Loss: {val_loss}")
            writer.add_scalar("val/loss", val_loss, epoch)
            if args.loss_sigma_correction:
                writer.add_scalar("val/tgt_nll", np.mean(tgt_nlls), epoch)
                writer.add_scalar("val/mat_nll", np.mean(mat_nlls), epoch)
                writer.add_scalar("val/tgt_mse", np.mean(tgt_mses), epoch)
                writer.add_scalar("val/mat_mse", np.mean(mat_mses), epoch)

    # loss parameters

    # list all parameters with their names
    loss_params = [(name, param) for name, param in criterion.named_parameters()]
    # log all parameters
    for name, param in loss_params:
        logger.info(f"{name}: {param}")

    writer.close()

    # Save model
    torch.save(model.state_dict(), os.path.join(args.save_path, args.name, "model.pt"))
    torch.save(
        criterion.state_dict(), os.path.join(args.save_path, args.name, "criterion.pt")
    )
