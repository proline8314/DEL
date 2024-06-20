import argparse
import gc
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import tensorboard as tb
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.chembl_dataset import FULL_NAME_DICT, ChemBLActivityDataset
from datasets.graph_dataset import GraphDataset
from datasets.lmdb_dataset import LMDBDataset
from losses.zip_loss import CorrectedZIPLoss, ZIPLoss
from models.ref_net import DELRefDecoder, DELRefEncoder, DELRefNet
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=25)
    parser.add_argument(
        "--save_path",
        type=str,
        default="/data02/gtguo/DEL/data/weights/refnet_nhop_distinct/",
    )
    parser.add_argument(
        "--output_save_path",
        type=str,
        default="/data02/gtguo/DEL/data/temp/nhop_distinct/",
    )
    parser.add_argument("--update_loss", action="store_true")
    # dataset
    parser.add_argument("--target_name", type=str, default="ca9")
    parser.add_argument("--fp_size", type=int, default=2048)
    parser.add_argument("--forced_reload", action="store_true")
    parser.add_argument("--nhop_split_num", type=int, default=3)

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
    parser.add_argument("--loss_sigma_correction", action="store_true")

    # record
    parser.add_argument(
        "--record_path",
        type=str,
        default="/data02/gtguo/DEL/data/records/refnet_nhop_distinct/",
    )

    # scheduler
    parser.add_argument("--lr_schedule", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--decay", type=float, default=0.1)
    parser.add_argument("--warmup_lr", type=float, default=5e-3)

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

    # Datasets

    def get_idx_array(dataset):
        idx_array = []
        for i in tqdm(range(len(dataset))):
            idx_array.append(dataset[i].mol_id.numpy())
        idx_array = np.array(idx_array)
        return idx_array

    def random_split_idx_about_nhop(idx_array, split_num=args.nhop_split_num):
        idxs_on_dim_1 = np.unique(idx_array[:, 1])
        idxs_on_dim_2 = np.unique(idx_array[:, 2])
        np.random.shuffle(idxs_on_dim_1)
        np.random.shuffle(idxs_on_dim_2)
        splits_dim_1 = np.array_split(idxs_on_dim_1, split_num)
        splits_dim_2 = np.array_split(idxs_on_dim_2, split_num)

        # combine splits and get nhop_split_num ** 2 set of nhop idx
        splits = []
        for idx_1 in range(split_num):
            for idx_2 in range(split_num):
                test_split_1 = splits_dim_1[idx_1]
                test_split_2 = splits_dim_2[idx_2]
                is_two_hop = np.logical_and(
                    np.isin(idx_array[:, 1], test_split_1),
                    np.isin(idx_array[:, 2], test_split_2),
                )
                is_one_hop = np.logical_xor(
                    np.isin(idx_array[:, 1], test_split_1),
                    np.isin(idx_array[:, 2], test_split_2),
                )
                is_zero_hop = np.logical_and(
                    np.logical_not(np.isin(idx_array[:, 1], test_split_1)),
                    np.logical_not(np.isin(idx_array[:, 2], test_split_2)),
                )

                logger.info(f"Zero-hop size: {np.sum(is_zero_hop)}")
                logger.info(f"One-hop size: {np.sum(is_one_hop)}")
                logger.info(f"Two-hop size: {np.sum(is_two_hop)}")

                assert (
                    np.logical_and(
                        np.logical_and(is_zero_hop, is_one_hop), is_two_hop
                    ).all()
                    == False
                )
                assert (
                    np.logical_or(
                        np.logical_or(is_zero_hop, is_one_hop), is_two_hop
                    ).all()
                    == True
                )

                splits.append(
                    (
                        is_zero_hop,
                        is_one_hop,
                        is_two_hop,
                        idx_array[is_zero_hop],
                        idx_array[is_one_hop],
                        idx_array[is_two_hop],
                    )
                )
        return splits
    """
    del_dataset = GraphDataset(
        forced_reload=args.forced_reload,
        target_name=args.target_name,
        fpsize=args.fp_size,
    )
    """
    del_dataset = LMDBDataset.readonly_raw(GraphDataset.DATASET_DIR, GraphDataset.name_dict[args.target_name])

    idx_array = get_idx_array(del_dataset)
    n_hop_splits = random_split_idx_about_nhop(idx_array)

    for num_split, n_hop_split in enumerate(n_hop_splits):
        if num_split == 0:
            continue

        logger.info(f"Split {num_split}")

        logger.info(f"ref count: {sys.getrefcount(del_dataset)}")

        del(del_dataset)

        logger.info(f"gc.collect() count: {gc.collect()}")

        logging.info("Refreshing dataset") # ! bug fix

        del_dataset = LMDBDataset.readonly_raw(GraphDataset.DATASET_DIR, GraphDataset.name_dict[args.target_name])

        is_zero_hop, is_one_hop, is_two_hop, zero_hop_idx, one_hop_idx, two_hop_idx = (
            n_hop_split
        )

        zero_hop_dataset = del_dataset[is_zero_hop]
        one_hop_dataset = del_dataset[is_one_hop]
        two_hop_dataset = del_dataset[is_two_hop]

        logger.info(f"Zero-hop dataset size: {len(zero_hop_dataset)}")
        logger.info(f"One-hop dataset size: {len(one_hop_dataset)}")
        logger.info(f"Two-hop dataset size: {len(two_hop_dataset)}")

        train_dataset, val_dataset = train_test_split(
            zero_hop_dataset,
            train_size=args.train_size,
            test_size=args.valid_size,
            random_state=seed,
        )

        # DataLoaders
        del_train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        del_val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # Set up tensorboard
        tb_path = os.path.join(
            os.path.dirname(__file__),
            os.path.join(args.record_path, f"{args.name}_{num_split}"),
        )
        if not os.path.exists(tb_path):
            os.makedirs(tb_path)
        writer = SummaryWriter(tb_path)

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
        logger.info(
            f"Model has {sum(p.numel() for p in model.parameters())} parameters"
        )

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
        def lambda_lr(
            epoch, total_epoch, warmup_lr, min_lr, warmup_ratio=0.1, decay=0.1
        ):
            assert warmup_ratio < 1
            assert decay < 1
            if epoch / total_epoch < warmup_ratio:
                return min_lr + (warmup_lr - min_lr) * epoch / (
                    total_epoch * warmup_ratio
                )
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
                y = torch.cat(
                    (
                        data.y_target.view(data.batch_size, args.target_size),
                        data.y_matrix.view(data.batch_size, args.matrix_size),
                    ),
                    dim=1,
                )
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                if i % args.log_interval == 0:
                    logger.info(
                        f"Epoch {epoch}, Iteration {i}, Train Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}"
                    )
                    writer.add_scalar(
                        "train/loss", loss.item(), epoch * len(del_train_loader) + i
                    )
                    if args.loss_sigma_correction:
                        _, tgt_nll, mat_nll, tgt_mse, mat_mse = criterion(
                            output, y, return_loss=True
                        )
                        writer.add_scalar(
                            "train/tgt_nll",
                            tgt_nll.item(),
                            epoch * len(del_train_loader) + i,
                        )
                        writer.add_scalar(
                            "train/mat_nll",
                            mat_nll.item(),
                            epoch * len(del_train_loader) + i,
                        )
                        writer.add_scalar(
                            "train/tgt_mse",
                            tgt_mse.item(),
                            epoch * len(del_train_loader) + i,
                        )
                        writer.add_scalar(
                            "train/mat_mse",
                            mat_mse.item(),
                            epoch * len(del_train_loader) + i,
                        )
                    writer.add_scalar(
                        "train/lr",
                        optimizer.param_groups[0]["lr"],
                        epoch * len(del_train_loader) + i,
                    )

            if args.lr_schedule:
                scheduler.step()
            """
            if epoch % args.save_interval == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.save_path, args.name, f"model_{epoch}.pt"),
                )
            """

            model.eval()
            with torch.no_grad():
                losses = []
                if args.loss_sigma_correction:
                    tgt_nlls = []
                    mat_nlls = []
                    tgt_mses = []
                    mat_mses = []
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
                    if args.loss_sigma_correction:
                        loss, tgt_nll, mat_nll, tgt_mse, mat_mse = criterion(
                            output,
                            torch.cat(
                                (
                                    data.y_target.view(
                                        data.batch_size, args.target_size
                                    ),
                                    data.y_matrix.view(
                                        data.batch_size, args.matrix_size
                                    ),
                                ),
                                dim=1,
                            ),
                            return_loss=True,
                        )
                        losses.append(loss.item())
                        tgt_nlls.append(tgt_nll.item())
                        mat_nlls.append(mat_nll.item())
                        tgt_mses.append(tgt_mse.item())
                        mat_mses.append(mat_mse.item())
                    else:
                        loss = criterion(
                            output,
                            torch.cat(
                                (
                                    data.y_target.view(
                                        data.batch_size, args.target_size
                                    ),
                                    data.y_matrix.view(
                                        data.batch_size, args.matrix_size
                                    ),
                                ),
                                dim=1,
                            ),
                        )
                        losses.append(loss.item())
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
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        torch.save(
            model.state_dict(),
            os.path.join(args.save_path, f"{args.name}_{num_split}.pt"),
        )

        # eval on n-hop dataset
        model.eval()
        zero_hop_loader = DataLoader(
            zero_hop_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        one_hop_loader = DataLoader(
            one_hop_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        two_hop_loader = DataLoader(
            two_hop_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        target_lambda_zero_hop = []
        target_lambda_one_hop = []
        target_lambda_two_hop = []

        with torch.no_grad():
            for loader, target_lambda in zip(
                [zero_hop_loader, one_hop_loader, two_hop_loader],
                [target_lambda_zero_hop, target_lambda_one_hop, target_lambda_two_hop],
            ):
                for i, data in tqdm(enumerate(loader)):
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
                    target_lambda.append(output[:, 0].detach().cpu())
                # target_lambda = torch.cat(target_lambda, dim=0).numpy()

        target_lambda_zero_hop = torch.cat(target_lambda_zero_hop, dim=0).numpy()
        target_lambda_one_hop = torch.cat(target_lambda_one_hop, dim=0).numpy()
        target_lambda_two_hop = torch.cat(target_lambda_two_hop, dim=0).numpy()

        # save target_lambda

        if not os.path.exists(args.output_save_path):
            os.makedirs(args.output_save_path)

        with open(
            os.path.join(
                args.output_save_path, f"{args.name}_zero_hop_{num_split}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(target_lambda_zero_hop, f)
        with open(
            os.path.join(args.output_save_path, f"{args.name}_one_hop_{num_split}.pkl"),
            "wb",
        ) as f:
            pickle.dump(target_lambda_one_hop, f)
        with open(
            os.path.join(args.output_save_path, f"{args.name}_two_hop_{num_split}.pkl"),
            "wb",
        ) as f:
            pickle.dump(target_lambda_two_hop, f)
        # dump n_hop_split_idxes
        with open(
            os.path.join(
                args.output_save_path, f"{args.name}_zero_hop_idx_{num_split}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(zero_hop_idx, f)
        with open(
            os.path.join(
                args.output_save_path, f"{args.name}_one_hop_idx_{num_split}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(one_hop_idx, f)
        with open(
            os.path.join(
                args.output_save_path, f"{args.name}_two_hop_idx_{num_split}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(two_hop_idx, f)
        # dump is in n_hop
        with open(
            os.path.join(
                args.output_save_path, f"{args.name}_is_zero_hop_{num_split}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(is_zero_hop, f)
        with open(
            os.path.join(
                args.output_save_path, f"{args.name}_is_one_hop_{num_split}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(is_one_hop, f)
        with open(
            os.path.join(
                args.output_save_path, f"{args.name}_is_two_hop_{num_split}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(is_two_hop, f)