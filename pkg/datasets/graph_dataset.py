import os
import sys
from functools import lru_cache

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.SaltRemover import SaltRemover
from torch_geometric.data import Data

sys.path.append("/data02/gtguo/DEL/pkg")
from ..utils.mol_feat import (get_edge_features, get_edge_index,
                              get_mol_graph_attr, get_node_features)
from .lmdb_dataset import LMDBDataset

DATASET_DIR = "/data02/gtguo/DEL/data/dataset/acs.jcim.2c01608"
RAW_FILE_NAME = "features.lmdb"
PROCESSED_FILE_NAME = "graph_dataset.lmdb"


class GraphDataset(LMDBDataset):
    # TODO (done): take survey in pyg dataloader to support for heterogeneous dict dataset
    DATASET_DIR = "/data02/gtguo/DEL/data/dataset/acs.jcim.2c01608"
    name_dict = {
        "ca9": "graph_dataset.lmdb",
        "hrp": "graph_dataset_hrp.lmdb",
    }
    def __init__(
        self,
        forced_reload: bool = False,
        target_name: str = "ca9",
        fpr: int = 2,
        fpsize: int = 2048,
    ):
        # TODO : support for different raw datasets
        assert target_name in ("ca9", "hrp")
        self.target_name = target_name
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=fpr, fpSize=fpsize
        )
        self.salt_remover = SaltRemover()

        DATASET_DIR = "/data02/gtguo/DEL/data/dataset/acs.jcim.2c01608"
        RAW_FILE_NAME = "features.lmdb"
        PROCESSED_FILE_NAME = GraphDataset.name_dict[target_name]
        super(GraphDataset, self).__init__(
            raw_dir=DATASET_DIR,
            raw_fname=RAW_FILE_NAME,
            processed_dir=DATASET_DIR,
            processed_fname=PROCESSED_FILE_NAME,
            forced_process=forced_reload,
        )

    @lru_cache(maxsize=16)
    def get_fingerprint(self, smiles: str) -> torch.Tensor:
        if not isinstance(smiles, str):
            smiles = ""
        mol = Chem.MolFromSmiles(smiles)
        mol = self.salt_remover.StripMol(mol)
        fp = np.array(self.mfpgen.GetFingerprint(mol), dtype=float)
        return fp

    def process(self, sample):
        mol = sample["mol"]
        mol_graph = sample["mol_graph"]
        y_matrix = torch.tensor(sample[f"{self.target_name}_matrix"], dtype=torch.int64)
        y_target = torch.tensor(sample[f"{self.target_name}_target"], dtype=torch.int64)
        bbsmiles_list = [sample[f"BB{i}_smiles"] for i in range(1, 4)]

        node_features = get_node_features(mol)
        edge_features = get_edge_features(mol)
        edge_index = get_edge_index(mol)
        node_bbidx = get_mol_graph_attr(mol, mol_graph, "motif")
        node_dist = get_mol_graph_attr(mol, mol_graph, "topo_dist")
        bbfp = np.array([self.get_fingerprint(smiles) for smiles in bbsmiles_list])
        bbfp = torch.tensor(bbfp, dtype=torch.float)

        mol_id = torch.tensor([int(sample[f"BB{i}_id"]) for i in range(1, 4)], dtype=torch.int64)

        # * Unstable: append topological info to node_feats
        # * node_features = torch.cat((node_features, node_bbidx.view(-1, 1), node_dist.view(-1, 1)), dim=1)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
        data.y_matrix = y_matrix
        data.y_target = y_target
        data.node_bbidx = node_bbidx
        data.node_dist = node_dist
        data.bbfp = bbfp
        data.mol_id = mol_id
        return data
