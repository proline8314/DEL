import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets.lmdb_dataset import LMDBDataset


class TargetDataset:
    def __init__(self, dataset, target_name):
        self.dataset = dataset
        self.target_name = target_name

    def __getitem__(self, idx):
        return self.dataset[idx]["readout"][self.target_name]

    def __len__(self):
        return len(self.dataset)


PI = 3.14159265358979323846


def calculate_minimun_loss(sample: torch.Tensor) -> float:
    loss = torch.mean(torch.log(2 * PI * sample + 1)) / 2
    return loss.item()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Calculate theoretical minimum loss")

    parser.add_argument(
        "--dataset_fpath", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--target_name", type=str, required=True)

    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    logger.info(f"Loading dataset from {args.dataset_fpath}")
    dataset = LMDBDataset.readonly_raw(*os.path.split(args.dataset_fpath))

    target_dataset = TargetDataset(dataset, args.target_name)
    target_dataloader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    readout_target_loss = []
    readout_control_loss = []

    logger.info("Calculating minimum loss")
    for batch in tqdm(target_dataloader):
        target_data = batch["target"]
        control_data = batch["control"]

        readout_target_loss.append(calculate_minimun_loss(target_data))
        readout_control_loss.append(calculate_minimun_loss(control_data))

    readout_target_loss = np.mean(readout_target_loss)
    readout_control_loss = np.mean(readout_control_loss)

    logger.info(f"Readout target loss: {readout_target_loss}")
    logger.info(f"Readout control loss: {readout_control_loss}")
