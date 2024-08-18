import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import tqdm

from ..datasets.lmdb_dataset import LMDBDataset
from ..utils.utils import standardize_smiles

readout_path = "E:/Research/del/data/raw/003-CA-2/2022CAS_DEL_all_exp.csv"
smiles_path = "E:/Research/del/data/raw/003-CA-2/CAS-DEL_smiles.csv"
bb_smiles_path = "E:/Research/del/data/raw/003-CA-2/output_100.txt"
output_path = "E:/Research/del/data/lmdb/003_CA2.lmdb"

# CodeA,CodeB,CodeC,Exp-B01,Exp-P01,Exp-P02,Exp-P03,Exp-A01,Exp-A02,Exp-A03,Exp-OA01,Exp-OA02,Exp-OA03

readout = pd.read_csv(readout_path)[
    [
        "CodeA",
        "CodeB",
        "CodeC",
        "Exp-B01",
        "Exp-P01",
        "Exp-P02",
        "Exp-P03",
        "Exp-A01",
        "Exp-A02",
        "Exp-A03",
        "Exp-OA01",
        "Exp-OA02",
        "Exp-OA03",
    ]
]
smiles = pd.read_csv(smiles_path)[["smiles", "CodeA", "CodeB", "CodeC"]]

with open(bb_smiles_path, "r") as f:
    lines = f.readlines()
    bb_smiles = [line.strip() for line in lines]
    codeA_idx, codeB_idx, codeC_idx = (
        bb_smiles.index("CodeA:"),
        bb_smiles.index("CodeB:"),
        bb_smiles.index("CodeC:"),
    )

    bb1_smiles = bb_smiles[codeA_idx + 1 : codeB_idx]
    bb1_smiles = [tup.split(":") for tup in bb1_smiles if ":" in tup]
    bb1_smiles = {int(tup[0]) + 1: tup[1].strip() for tup in bb1_smiles}
    bb1_smiles = {idx: standardize_smiles(smiles) for idx, smiles in bb1_smiles.items()}

    bb2_smiles = bb_smiles[codeB_idx + 1 : codeC_idx]
    bb2_smiles = [tup.split(":") for tup in bb2_smiles if ":" in tup]
    bb2_smiles = {int(tup[0]) + 1: tup[1].strip() for tup in bb2_smiles}
    bb2_smiles = {idx: standardize_smiles(smiles) for idx, smiles in bb2_smiles.items()}

    bb3_smiles = bb_smiles[codeC_idx + 1 :]
    bb3_smiles = [tup.split(":") for tup in bb3_smiles if ":" in tup]
    bb3_smiles = {int(tup[0]) + 1: tup[1].strip() for tup in bb3_smiles}
    bb3_smiles = {idx: standardize_smiles(smiles) for idx, smiles in bb3_smiles.items()}

dataset = pd.merge(readout, smiles, on=["CodeA", "CodeB", "CodeC"])
dataset["bb1_smiles"] = dataset["CodeA"].map(bb1_smiles)
dataset["bb2_smiles"] = dataset["CodeB"].map(bb2_smiles)
dataset["bb3_smiles"] = dataset["CodeC"].map(bb3_smiles)


def line_to_dict_sample(line: pd.Series) -> Dict[str, Any]:
    return {
        "smiles": line["smiles"],
        "readout": {
            "CA2": {
                "target": np.array(
                    [
                        line["Exp-P01"],
                        line["Exp-P02"],
                        line["Exp-P03"],
                    ],
                    dtype=float,
                ),
                "control": np.array(line["Exp-B01"], dtype=float),
            },
            "CA12": {
                "target": np.array(
                    [
                        line["Exp-A01"],
                        line["Exp-A02"],
                        line["Exp-A03"],
                    ],
                    dtype=float,
                ),
                "control": np.array(line["Exp-B01"], dtype=float),
            },
            "OCA12": {
                "target": np.array(
                    [
                        line["Exp-OA01"],
                        line["Exp-OA02"],
                        line["Exp-OA03"],
                    ],
                    dtype=float,
                ),
                "control": np.array(line["Exp-B01"], dtype=float),
            },
        },
        "bb1_smiles": line["bb1_smiles"],
        "bb2_smiles": line["bb2_smiles"],
        "bb3_smiles": line["bb3_smiles"],
    }


dataset = [line_to_dict_sample(dataset.iloc[i]) for i in tqdm.tqdm(range(len(dataset)))]
print(dataset[0])
dataset = LMDBDataset.static_from_others(dataset, *os.path.split(output_path), map_size=4294967296*2)
print(dataset[0])