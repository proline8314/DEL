import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import tqdm

from ..datasets.lmdb_dataset import LMDBDataset
from ..utils.utils import standardize_smiles

fpath = "E:/Research/del/data/raw/002-CAIX/ja9b01203_si_002.xlsx"
output_path = "E:/Research/del/data/lmdb/002_CAIX.lmdb"

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


def index_to_dict_sample(index: int) -> Dict[str, Any]:
    row = dataset.iloc[index]
    return {
        "smiles": row["smiles"],
        "readout": {
            "CA9": {
                "target": np.array(
                    [
                        row["ca9_exp_r1"],
                        row["ca9_exp_r2"],
                        row["ca9_exp_r3"],
                        row["ca9_exp_r4"],
                    ],
                    dtype=float,
                ),
                "control": np.array(
                    [row["ca9_beads_r1"], row["ca9_beads_r2"]], dtype=float
                ),
            },
            "HRP": {
                "target": np.array([row["hrp_exp_r1"], row["hrp_exp_r2"]], dtype=float),
                "control": np.array(
                    [
                        row["hrp_beads_r1"],
                        row["hrp_beads_r2"],
                        row["hrp_beads_r3"],
                        row["hrp_beads_r4"],
                    ],
                    dtype=float,
                ),
            },
        },
        "bb1_smiles": row["smiles_scaffold"],
        "bb2_smiles": row["smiles_bb1"],
        "bb3_smiles": row["smiles_bb2"],
    }


dataset = [index_to_dict_sample(index) for index in tqdm.tqdm(range(len(dataset)))]
# save dataset
print(dataset[0])
dataset = LMDBDataset.static_from_others(dataset, *os.path.split(output_path), forced_process=True, map_size=1024**3 * 16)
print(dataset[0])