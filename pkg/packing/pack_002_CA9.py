import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import tqdm

from ..datasets.lmdb_dataset import LMDBDataset
from ..utils.utils import standardize_smiles

fpath = "E:/Research/del/data/raw/002-CAIX/ja9b01203_si_002.xlsx"

bb_smiles = pd.read_excel(fpath, sheet_name="D2")
mol_smiles = pd.read_excel(fpath, sheet_name="D5")
readouts = pd.read_excel(fpath, sheet_name="D6")

bb_smiles = bb_smiles[["position", "index", "smiles"]]

scaffold = bb_smiles[bb_smiles["position"] == "scaffold"]
scaffold = scaffold.rename(columns={"index": "scaffold"})
scaffold = scaffold[["scaffold", "smiles"]]

bb1 = bb_smiles[bb_smiles["position"] == "BB1"]
bb1 = bb1.rename(columns={"index": "BB1"})
bb1 = bb1[["BB1", "smiles"]]

bb2 = bb_smiles[bb_smiles["position"] == "BB2"]
bb2 = bb2.rename(columns={"index": "BB2"})
bb2 = bb2[["BB2", "smiles"]]

mol_smiles = mol_smiles[["cpd_id", "smiles"]]
readouts = readouts[
    [
        "cpd_id",
        "scaffold",
        "BB1",
        "BB2",
        "hrp_beads_r1",
        "hrp_beads_r2",
        "hrp_beads_r3",
        "hrp_beads_r4",
        "hrp_exp_r1",
        "hrp_exp_r2",
        "ca9_beads_r1",
        "ca9_beads_r2",
        "ca9_exp_r1",
        "ca9_exp_r2",
        "ca9_exp_r3",
        "ca9_exp_r4",
    ]
]

# merge
dataset = pd.merge(readouts, mol_smiles, on="cpd_id")
dataset = pd.merge(dataset, scaffold, on="scaffold", suffixes=("", "_scaffold"))
dataset = pd.merge(dataset, bb1, on="BB1", suffixes=("", "_bb1"))
dataset = pd.merge(dataset, bb2, on="BB2", suffixes=("", "_bb2"))
print(dataset.head())
print(dataset.columns)
print(dataset.shape)
print(dataset.iloc[0])