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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool as gap
from tqdm import tqdm
from utils.mol_feat import process_to_pyg_data

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up arguments
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--name", type=str, default="ca9_vs_10")
    parser.add_argument("--weight_name", type=str, default="ca9_full_real")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument(
        "--save_path", type=str, default="/data02/gtguo/DEL/data/weights/refnet_ca9_vs/"
    )
    parser.add_argument(
        "--load_path", type=str, default="/data02/gtguo/DEL/data/weights/refnet/"
    )
    parser.add_argument("--load_weight", action="store_true")

    # dataset
    parser.add_argument("--target_name", type=str, default="ca9")
    parser.add_argument("--fp_size", type=int, default=2048)
    parser.add_argument("--inactive_multiplier", type=int, default=10)
    parser.add_argument("--active_score_threshold", type=float, default=6.0)

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
        return {"pyg_data": pyg_data, "activity": activity, "smiles": smiles}

    chembl_dataset = ChemBLActivityDataset(
        FULL_NAME_DICT[args.target_name], update_target=False
    )
    active_dataset, inactive_dataset = chembl_dataset.split_with_condition(
        lambda is_hit: is_hit == 1,
        (("Carbonic anhydrase IX", "is_hit"),),
    )
    active_dataset = LMDBDataset.update_process_fn(
        process_fn=process
    ).dynamic_from_others(active_dataset)

    real_active_dataset, false_active_dataset = active_dataset.split_with_condition(
        lambda activity: activity > args.active_score_threshold,
        ("activity",),
    )

    active_dataset = real_active_dataset
    if len(false_active_dataset) < len(real_active_dataset) * args.inactive_multiplier:
        len_inactive_dataset = len(
            real_active_dataset
        ) * args.inactive_multiplier - len(false_active_dataset)
        inactive_dataset = train_test_split(
            inactive_dataset, train_size=len_inactive_dataset, random_state=seed
        )[0]
        inactive_dataset = LMDBDataset.update_process_fn(
            process_fn=process
        ).dynamic_from_others(inactive_dataset)
        inactive_dataset = inactive_dataset + false_active_dataset

        logger.info(
            f"Selected {len_inactive_dataset} inactive samples from the original dataset"
        )
    else:
        inactive_dataset = train_test_split(
            false_active_dataset,
            train_size=len(real_active_dataset) * args.inactive_multiplier,
            random_state=seed,
        )[0]

    # inactive_dataset = LMDBDataset.static_from_others(inactive_dataset, processed_dir="/data02/gtguo/DEL/data/dataset/chembl", processed_fname="ca9_inactive_thr6.lmdb")
    # raise ValueError

    logger.info(f"Active dataset has {len(active_dataset)} samples")
    logger.info(f"Dataset data example: {active_dataset[0]}")

    # filter large molecules
    """
    temp_dataset = []
    for sample in inactive_dataset:
        if sample["pyg_data"].x.size(0) < 50:
            temp_dataset.append(sample)
    inactive_dataset = LMDBDataset.dynamic_from_others(temp_dataset)
    """
    logger.info(f"Inactive dataset has {len(inactive_dataset)} samples")

    # DataLoaders
    active_data_loader = DataLoader(
        active_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    inactive_data_loader = DataLoader(
        inactive_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Load model
    if args.load_path and args.load_weight:
        logger.info(f"Load model from {os.path.join(args.load_path, args.weight_name)}")
        model.load_state_dict(
            torch.load(os.path.join(args.load_path, args.weight_name, "model.pt"), map_location=device),
            strict=False,
        )

    # Test
    model.eval()
    output_data = []

    for batch in tqdm(active_data_loader):
        data = batch["pyg_data"].to(device)
        output = model(
            data.x, data.edge_index, data.edge_attr, None, data.batch, None, None
        )
        output = output.detach().cpu()
        output_data.append(output)
    logger.info("Done active")

    for batch in tqdm(inactive_data_loader):
        data = batch["pyg_data"].to(device)
        output = model(
            data.x, data.edge_index, data.edge_attr, None, data.batch, None, None
        )
        output = output.detach().cpu()
        output_data.append(output)
    logger.info("Done inactive")

    output_data = torch.cat(output_data, dim=0).numpy()
    is_active = np.concatenate(
        [np.ones(len(active_dataset)), np.zeros(len(inactive_dataset))]
    )[: len(output_data)]
    active_ratio = len(active_dataset) / len(output_data)
    logger.info(f"Active ratio: {active_ratio}")
    target_lambda, matrix_lambda = output_data[:, 0], output_data[:, 1]

    def enrichment_factor(y_true, y_pred, cutoff=0.1):
        y_true = y_true[y_pred.argsort()[::-1]]
        return y_true[: int(len(y_true) * cutoff)].mean() / y_true.mean()

    # use target_lambda to predict activity
    logger.info("Use target_lambda to predict activity")
    logger.info(
        f"Enrichment factor 0.1%: {enrichment_factor(is_active, target_lambda, 0.001)}"
    )
    logger.info(
        f"Enrichment factor 1%: {enrichment_factor(is_active, target_lambda, 0.01)}"
    )
    logger.info(
        f"Enrichment factor 5%: {enrichment_factor(is_active, target_lambda, 0.05)}"
    )
    logger.info(f"Enrichment factor 10%: {enrichment_factor(is_active, target_lambda)}")

    logger.info(f"ROC AUC: {roc_auc_score(is_active, target_lambda)}")

    # use matrix_lambda to predict activity
    logger.info("Use matrix_lambda to predict activity")
    logger.info(
        f"Enrichment factor 0.1%: {enrichment_factor(is_active, matrix_lambda, 0.001)}"
    )
    logger.info(
        f"Enrichment factor 1%: {enrichment_factor(is_active, matrix_lambda, 0.01)}"
    )
    logger.info(
        f"Enrichment factor 5%: {enrichment_factor(is_active, matrix_lambda, 0.05)}"
    )
    logger.info(f"Enrichment factor 10%: {enrichment_factor(is_active, matrix_lambda)}")

    logger.info(f"ROC AUC: {roc_auc_score(is_active, matrix_lambda)}")

    # use negative matrix_lambda to predict activity
    logger.info("Use negative matrix_lambda to predict activity")
    logger.info(
        f"Enrichment factor 0.1%: {enrichment_factor(is_active, -matrix_lambda, 0.001)}"
    )
    logger.info(
        f"Enrichment factor 1%: {enrichment_factor(is_active, -matrix_lambda, 0.01)}"
    )
    logger.info(
        f"Enrichment factor 5%: {enrichment_factor(is_active, -matrix_lambda, 0.05)}"
    )
    logger.info(
        f"Enrichment factor 10%: {enrichment_factor(is_active, -matrix_lambda)}"
    )

    logger.info(f"ROC AUC: {roc_auc_score(is_active, -matrix_lambda)}")

    # use diff enrichment factor to predict activity
    logger.info("Use diff to predict activity")
    logger.info(
        f"Enrichment factor 0.1%: {enrichment_factor(is_active, target_lambda - matrix_lambda, 0.001)}"
    )
    logger.info(
        f"Enrichment factor 1%: {enrichment_factor(is_active, target_lambda - matrix_lambda, 0.01)}"
    )
    logger.info(
        f"Enrichment factor 5%: {enrichment_factor(is_active, target_lambda - matrix_lambda, 0.05)}"
    )
    logger.info(
        f"Enrichment factor 10%: {enrichment_factor(is_active, target_lambda - matrix_lambda)}"
    )

    logger.info(f"ROC AUC: {roc_auc_score(is_active, target_lambda - matrix_lambda)}")

    # use ratio enrichment factor to predict activity
    logger.info("Use ratio to predict activity")
    logger.info(
        f"Enrichment factor 0.1%: {enrichment_factor(is_active, target_lambda / matrix_lambda, 0.001)}"
    )
    logger.info(
        f"Enrichment factor 1%: {enrichment_factor(is_active, (target_lambda + 1) / (matrix_lambda + 1), 0.01)}"
    )
    logger.info(
        f"Enrichment factor 5%: {enrichment_factor(is_active, (target_lambda + 1) / (matrix_lambda + 1), 0.05)}"
    )
    logger.info(
        f"Enrichment factor 10%: {enrichment_factor(is_active, (target_lambda + 1) / (matrix_lambda + 1))}"
    )
    logger.info(
        f"ROC AUC: {roc_auc_score(is_active, (target_lambda + 1) / (matrix_lambda + 1))}"
    )

    # use modulated diff enrichment factor to predict activity (use logistic regression to get the weight)
    logger.info("Use modulated diff to predict activity")
    X = output_data
    y = is_active
    train_sizes = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    for train_size in train_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=seed
        )
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:, 1]
        logger.info(f"Train size: {train_size}")
        logger.info(
            f"Enrichment factor 0.1%: {enrichment_factor(y_test, y_pred, 0.001)}"
        )
        logger.info(f"Enrichment factor 1%: {enrichment_factor(y_test, y_pred, 0.01)}")
        logger.info(f"Enrichment factor 5%: {enrichment_factor(y_test, y_pred, 0.05)}")
        logger.info(f"Enrichment factor 10%: {enrichment_factor(y_test, y_pred)}")
        logger.info(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
