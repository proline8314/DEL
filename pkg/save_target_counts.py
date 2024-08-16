import argparse
import logging
import os
import pickle
import sys

import numpy as np
from datasets.graph_dataset import GraphDataset
from tqdm import tqdm

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up arguments
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--seed", type=int, default=4)

    # dataset
    parser.add_argument("--target_name", type=str, default="hrp")
    parser.add_argument("--fp_size", type=int, default=2048)
    parser.add_argument("--forced_reload", action="store_true")

    args = parser.parse_args()

    # print arguments to log
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    save_dir = f"/data02/gtguo/DEL/data/temp/counts/{args.target_name}"
    # Datasets
    del_dataset = GraphDataset(
        forced_reload=args.forced_reload,
        target_name=args.target_name,
        fpsize=args.fp_size,
    )

    def get_idx_array(dataset):
        idx_array = []
        for i in tqdm(range(len(dataset))):
            idx_array.append(dataset[i].mol_id.numpy())
        idx_array = np.array(idx_array)
        return idx_array

    dataset_length = len(del_dataset)
    logger.info(f"Dataset size: {len(del_dataset)}")
    idx_array = get_idx_array(del_dataset)
    logger.info(f"idx_array size: {idx_array.shape}")

    target_counts = []
    for i in tqdm(range(dataset_length)):
        target_counts.append(del_dataset[i].y_target.numpy())
    target_counts = np.array(target_counts)
    logger.info(f"target_counts size: {target_counts.shape}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(f"{save_dir}/target_counts.pkl", "wb") as f:
        pickle.dump(target_counts, f)
    with open(f"{save_dir}/idx_array.pkl", "wb") as f:
        pickle.dump(idx_array, f)
