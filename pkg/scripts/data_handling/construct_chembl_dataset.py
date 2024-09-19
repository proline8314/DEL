import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import rdkit
import tensorboard as tb
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from torch_geometric.nn import global_mean_pool as gap
from tqdm import tqdm

from ...datasets.chembl_dataset import FULL_NAME_DICT, ChemBLActivityDataset
from ...datasets.lmdb_dataset import LMDBDataset

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up arguments
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--data_path", type=str, default="/data03/gtguo/data/chembl/lmdb/chembl_activity.lmdb")
    parser.add_argument("--output_dir", type=str, default="/data03/gtguo/data/chembl/lmdb/target_hits/ca9")
    parser.add_argument("--target_name", type=str, default="ca9")
    parser.add_argument("--inactive_multiplier", type=int, default=10)
    parser.add_argument("--active_score_threshold", type=float, default=6.0)


    args = parser.parse_args()

    # print arguments to log
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # Set up random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Datasets
    def process(sample):
        smiles = sample["mol_structures"]["smiles"]
        activity = sample[FULL_NAME_DICT[args.target_name]]["activity"]
        return {"smiles": smiles, "activity": activity}

    chembl_dataset = ChemBLActivityDataset(
        FULL_NAME_DICT[args.target_name], update_target=False
    )
    active_dataset, inactive_dataset = chembl_dataset.split_with_condition(
        lambda is_hit: is_hit == 1,
        ((FULL_NAME_DICT[args.target_name], "is_hit"),),
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
        inactive_dataset = LMDBDataset.update_process_fn(process_fn=process).dynamic_from_others(
            inactive_dataset
        )
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

    active_dataset = LMDBDataset.static_from_others(active_dataset, args.output_dir, f"{args.target_name}_active_thr{args.active_score_threshold}.lmdb")
    inactive_dataset = LMDBDataset.static_from_others(inactive_dataset, args.output_dir, f"{args.target_name}_inactive_thr{args.active_score_threshold}.lmdb")

    logger.info(f"Active dataset has {len(active_dataset)} samples")
    logger.info(f"Inactive dataset has {len(inactive_dataset)} samples")
    logger.info(f"Dataset data example: {active_dataset[0]}")
