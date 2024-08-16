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
from datasets.chembl_dataset import FULL_NAME_DICT, ChemBLActivityDataset
from datasets.lmdb_dataset import LMDBDataset
from models.ref_net import DELRefDecoder, DELRefEncoder, DELRefNet
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from torch_geometric.nn import global_mean_pool as gap
from tqdm import tqdm

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up arguments
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="/data02/gtguo/DEL/data/dataset/ca9_ligands_small")
    parser.add_argument("--target_name", type=str, default="ca9")
    parser.add_argument("--fp_size", type=int, default=2048)
    parser.add_argument("--inactive_multiplier", type=int, default=10)
    parser.add_argument("--active_score_threshold", type=float, default=0.0)


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
    """
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
    """
    logger.info(f"Active dataset has {len(active_dataset)} samples")
    logger.info(f"Dataset data example: {active_dataset[0]}")

    """
    logger.info(f"Inactive dataset has {len(inactive_dataset)} samples")

    # save all ligands to mol2 files

    is_active = [1] * len(active_dataset) + [0] * len(inactive_dataset)
    is_active_small = []
    all_dataset = active_dataset + inactive_dataset
    for i, sample in enumerate(all_dataset):
        # reduce by 10
        if i % 10 != 0:
            continue
        smiles = sample["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        # add hydrogens
        mol = Chem.AddHs(mol)
        # generate 3D coordinates
        AllChem.EmbedMolecule(mol)
        # write to pdb file
        os.makedirs(f"{args.output_dir}/{i // 10}", exist_ok=True)
        Chem.MolToPDBFile(mol, f"{args.output_dir}/{i // 10}/ligand.pdb")
        is_active_small.append(is_active[i])

    # save active label
    np.save(f"{args.output_dir}/is_active.npy", np.array(is_active_small))
    """
