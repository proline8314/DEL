import os
import pickle

import lmdb
import networkx as nx
import numpy as np
import rdkit
import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


class GraphDataset(Dataset):
    def __init__(self, root="/data02/gtguo/DEL/data/dataset/acs.jcim.2c01608", transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.txn = lmdb.open(os.path.join(self.processed_dir, self.processed_file_names), subdir=False,
                            readonly=False,
                            lock=False,
                            readahead=False,
                            meminit=False,
                            max_readers=1,
                            map_size=1099511627776).begin()

    @property
    def raw_file_names(self):
        return "features.lmdb"

    @property
    def processed_file_names(self):
        return "features_data.lmdb"

    def download(self):
        pass

    def process(self):
        new_env = lmdb.open(os.path.join(self.processed_dir, "features_data.lmdb"), subdir=False,
                            readonly=False,
                            lock=False,
                            readahead=False,
                            meminit=False,
                            max_readers=1,
                            map_size=1099511627776)
        new_txn = new_env.begin(write=True)
        with lmdb.open(os.path.join(self.raw_dir, "features.lmdb"), subdir=False,
                       readonly=False,
                       lock=False,
                       readahead=False,
                       meminit=False,
                       max_readers=1,
                       map_size=1099511627776).begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))
            for idx in tqdm(range(len(self._keys))):
                data_dict = pickle.loads(txn.get(f"{idx}".encode()))
                node_feat = self._get_node_features(data_dict)
                edge_feat = self._get_edge_features(data_dict)
                edge_index = self._get_edge_index(data_dict)
                label = self._get_target(data_dict)

                data = Data(x=node_feat, edge_index=edge_index,
                            edge_attr=edge_feat, y=label)
                new_txn.put(
                    str(idx).encode(),
                    pickle.dumps(
                        data,
                        protocol=-1
                    )
                )
        new_txn.commit()
        new_env.close()

    def _get_node_features(self, data_dict):
        all_node_features = []
        mol = data_dict["mol"]
        mol_graph = data_dict["mol_graph"]
        for atom in mol.GetAtoms():
            node_feature = []
            node_feature.append(atom.GetAtomicNum())
            node_feature.append(atom.GetDegree())
            node_feature.append(atom.GetFormalCharge())
            node_feature.append(atom.GetHybridization())
            node_feature.append(atom.GetIsAromatic())

            idx = atom.GetIdx()
            node_feature.append(int(mol_graph.nodes[idx]["motif"]))
            node_feature.append(mol_graph.nodes[idx]["topo_dist"])

            all_node_features.append(node_feature)
        all_node_features = np.asarray(all_node_features)
        return torch.tensor(all_node_features, dtype=torch.float)

    def _get_edge_index(self, data_dict):
        mol = data_dict["mol"]
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        row, col = np.where(adj_matrix)
        coo = np.array(list(zip(row, col)))
        coo = np.reshape(coo, (2, -1))
        return torch.tensor(coo, dtype=torch.long)

    def _get_edge_features(self, data_dict):
        all_edge_feature = []
        mol = data_dict["mol"]
        for bond in mol.GetBonds():
            edge_feature = []
            edge_feature.append(bond.GetBondTypeAsDouble())
            edge_feature.append(bond.IsInRing())
            all_edge_feature.append(edge_feature)

        all_edge_feature = np.asarray(all_edge_feature)
        return torch.tensor(all_edge_feature, dtype=torch.float)

    def _get_target(self, data_dict):
        label = np.concatenate(
            (data_dict["ca9_target"], data_dict["ca9_matrix"]))
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        if not hasattr(self, "_keys"):
            self._keys = list(self.txn.cursor().iternext(values=False))
        return len(self._keys)

    def get(self, idx):
        enc_idx = str(idx).encode()
        data = self.txn.get(enc_idx)
        data = pickle.loads(data)
        return data

if __name__ == "__main__":
    dataset = GraphDataset()
    print(dataset[0].x)
    print(dataset[0].edge_attr)
    print(dataset[0].y)