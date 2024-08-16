import logging
import os
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import pandas as pd
import rdkit
from get_synthon_index import get_synthon_index_from_smiles
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
from utils import (clean_up_mol, find_mutual_substruct, get_mol_from_smiles,
                   get_standardized_smiles)

RDLogger.DisableLog("rdApp.*")

data_path = "/data02/gtguo/DEL/data/raw/003-CA-2/CAS-DEL_smiles.csv"
save_path = "/data02/gtguo/DEL/code/data_preprocess"

# Load data, header: ['smiles', 'CodeA', 'CodeB', 'CodeC']
syns_names = ["CodeA", "CodeB", "CodeC"]
data = pd.read_csv(data_path)


# Count the number of synthon for CodeA, CodeB, CodeC
def count_number(name):
    syns_idx = data[name].values
    syns_num = len(set(syns_idx))
    return syns_num


syns_num_A = count_number("CodeA")
syns_num_B = count_number("CodeB")
syns_num_C = count_number("CodeC")

print(f"Number of synthon for CodeA: {syns_num_A}")
print(f"Number of synthon for CodeB: {syns_num_B}")
print(f"Number of synthon for CodeC: {syns_num_C}")

names_counts_dict = {name: count_number(name) for name in syns_names}
counts_array = np.array([names_counts_dict[name] for name in syns_names])

# data["idx"] = (
#     data["CodeC"] + data["CodeB"] * syns_num_C + data["CodeA"] * syns_num_C * syns_num_B
# )
print(data.head())

# take mutual substructure of DEL molecules as synthon
num_trial_mols_per_syns = 50
num_process = 8
replace = False


def choose_trial_mols(name, syn_idx):
    assert name in syns_names
    names_to_choose = [n for n in syns_names if n != name]

    # choose num_trial_mols_per_syns synthon idx for names_to_choose
    choosed_idx = {
        n: np.random.choice(data[n].values, num_trial_mols_per_syns, replace=replace)
        for n in names_to_choose
    }
    choosed_idx[name] = np.ones(num_trial_mols_per_syns) * syn_idx

    # get synthon idxs, shape: (num_trial_mols_per_syns, 3)
    choosed_name_idx_list = np.array(
        [
            [choosed_idx[n][i] for i in range(num_trial_mols_per_syns)]
            for n in syns_names
        ]
    ).transpose()

    choosed_idx_list = (
        np.array([[syns_num_C * syns_num_B, syns_num_C, 1]])
        * (choosed_name_idx_list - 1)
    ).sum(axis=1)

    choosed_smiles_list = data.loc[choosed_idx_list, "smiles"].values

    return choosed_smiles_list, choosed_idx_list, choosed_name_idx_list


syn_smiles = {n: [""] * names_counts_dict[n] for n in syns_names}


@lru_cache(maxsize=4096)
def get_cleaned_mol(smiles):
    mol = get_mol_from_smiles(smiles)
    mol = clean_up_mol(mol)
    return mol


for name in syns_names:
    print(f"Processing {name}")

    def _handle_synthon(sidx):
        synthon = None
        smiles, _, _ = choose_trial_mols(name, sidx + 1)
        for i in range(num_trial_mols_per_syns):
            mol = get_cleaned_mol(smiles[i])
            if synthon is None:
                synthon = mol
            else:
                synthon = find_mutual_substruct(synthon, mol)
        synthon = Chem.AddHs(synthon, explicitOnly=True)
        return Chem.MolToSmiles(synthon)

    with Pool(num_process) as p:
        syn_smiles[name] = list(
            tqdm(
                p.imap(_handle_synthon, range(names_counts_dict[name])),
                total=names_counts_dict[name],
            )
        )

# Save synthon smiles
with open(os.path.join(save_path, f"output_{num_trial_mols_per_syns}.txt"), "w") as f:
    for name in syns_names:
        f.write(f"{name}:\n")
        for i, s in enumerate(syn_smiles[name]):
            f.write(f"{i}: {s}\n")
        f.write("\n")
