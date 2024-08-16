import os
import pickle

import lmdb
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

fpath = "/data02/gtguo/DEL/data/raw/002-CAIX/ja9b01203_si_002.xlsx"
bb_smiles_df = pd.read_excel(fpath, sheet_name="D2")
mol_smiles_df = pd.read_excel(fpath, sheet_name="D5")
counts_df = pd.read_excel(fpath, sheet_name="D6")

# TODO need further abstract


def filter_nan(s: str | float):
    return s if s != float("nan") else ""


def get_bb_smiles_dict(df: pd.DataFrame):
    """
    [{bb_id: bb_smiles}]
    """
    position_name = ("scaffold", "BB1", "BB2")
    bb_dict_pack = [{}, {}, {}]
    for i, name in enumerate(position_name):
        df_slice = df[df["position"] == name]
        slice_dict = df_slice.to_dict()
        index_dict, smiles_dict = slice_dict["index"], slice_dict["smiles"]
        for row_index, index in index_dict.items():
            bb_dict_pack[i][index] = filter_nan(smiles_dict[row_index])
    return bb_dict_pack


def get_bb_mol_dict(bb_smiles_dict: dict, addHs: bool = False):
    """
    [{bb_id: bb_mol}]
    """
    bb_mol_dict_pack = [{}, {}, {}]
    for i, bb_dict in enumerate(bb_smiles_dict):
        for bb_id, bb_smiles in bb_dict.items():
            bb_mol = Chem.MolFromSmiles(bb_smiles)
            if addHs:
                bb_mol = Chem.AddHs(bb_mol)
            bb_mol_dict_pack[i][bb_id] = bb_mol
    return bb_mol_dict_pack


def get_mol_smiles_dict(df: pd.DataFrame):
    """
    {cpd_id: mol_smiles}
    """
    cols_to_retain = ["cpd_id", "smiles"]
    df = df.loc[:, cols_to_retain]
    df_dict = df.to_dict()
    id_dict, smiles_dict = df_dict["cpd_id"], df_dict["smiles"]

    output_dict = {}
    for row_index, mol_id in id_dict.items():
        output_dict[mol_id] = smiles_dict[row_index]
    return output_dict


def handle_counts_df(df: pd.DataFrame):
    cols_to_retain = ["cpd_id", "scaffold", "BB1", "BB2", "hrp_beads_r1", "hrp_beads_r2", "hrp_beads_r3", "hrp_beads_r4",
                      "hrp_exp_r1", "hrp_exp_r2", "ca9_beads_r1", "ca9_beads_r2", "ca9_exp_r1", "ca9_exp_r2", "ca9_exp_r3", "ca9_exp_r4"]
    df = df.loc[:, cols_to_retain]
    output_dict = df.to_dict(orient="index")
    return output_dict


def pack_single_mol_info(bb_smiles_dict, mol_smiles_dict, data_dict: dict):
    """
    Pack the information of a single molecule into a dictionary.
    """
    output_dict = {}
    output_dict["BB1_id"] = data_dict["scaffold"]
    output_dict["BB2_id"] = data_dict["BB1"]
    output_dict["BB3_id"] = data_dict["BB2"]
    output_dict["mol_id"] = "mol_" + "_".join(map(lambda x: "0" * (3 - len(
        str(x))) + str(x), [data_dict["scaffold"], data_dict["BB1"], data_dict["BB2"]]))
    output_dict["BB1_smiles"] = filter_nan(
        bb_smiles_dict[0][data_dict["scaffold"]])
    output_dict["BB2_smiles"] = filter_nan(bb_smiles_dict[1][data_dict["BB1"]])
    output_dict["BB3_smiles"] = filter_nan(bb_smiles_dict[2][data_dict["BB2"]])
    output_dict["mol_smiles"] = mol_smiles_dict[data_dict["cpd_id"]]
    output_dict["hrp_matrix"] = np.array(
        [data_dict["hrp_beads_r1"], data_dict["hrp_beads_r2"], data_dict["hrp_beads_r3"], data_dict["hrp_beads_r4"]])
    output_dict["hrp_target"] = np.array(
        [data_dict["hrp_exp_r1"], data_dict["hrp_exp_r2"]])
    output_dict["ca9_matrix"] = np.array(
        [data_dict["ca9_beads_r1"], data_dict["ca9_beads_r2"]])
    output_dict["ca9_target"] = np.array(
        [data_dict["ca9_exp_r1"], data_dict["ca9_exp_r2"], data_dict["ca9_exp_r3"], data_dict["ca9_exp_r4"]])
    return output_dict


bb_smiles_dict = get_bb_smiles_dict(bb_smiles_df)
mol_smiles_dict = get_mol_smiles_dict(mol_smiles_df)
counts_dict = handle_counts_df(counts_df)

save_root_dir = "/data02/gtguo/DEL/data/dataset/acs.jcim.2c01608"

with lmdb.open("/data02/gtguo/DEL/data/dataset/acs.jcim.2c01608/counts.lmdb", subdir=False,
               readonly=False,
               lock=False,
               readahead=False,
               meminit=False,
               max_readers=1,
               map_size=1099511627776) as env:
    txn = env.begin(write=True)

    for i, (row_id, data_dict) in tqdm(enumerate(counts_dict.items())):
        mol_info = pack_single_mol_info(
            bb_smiles_dict, mol_smiles_dict, data_dict)
        txn.put(str(i).encode(), pickle.dumps(mol_info, protocol=-1))
    txn.commit()


with lmdb.open("/data02/gtguo/DEL/data/dataset/acs.jcim.2c01608/bbsmiles.lmdb", subdir=False,
               readonly=False,
               lock=False,
               readahead=False,
               meminit=False,
               max_readers=1,
               map_size=1099511627776) as env:
    txn = env.begin(write=True)
    txn.put("0".encode(), pickle.dumps(bb_smiles_dict, protocol=-1))
    txn.commit()
