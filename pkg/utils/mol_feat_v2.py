from functools import lru_cache
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import rdkit
import torch
from rdkit import Chem
from rdkit.Chem import rdchem, rdFingerprintGenerator, rdmolops
from rdkit.Chem.SaltRemover import SaltRemover
from torch_geometric.data import Data

from .utils import clean_up_mol, get_mol_from_smiles


def get_node_features(mol: Chem.Mol, **kwargs) -> torch.Tensor:
    node_features = []
    maximum_num_atoms = 118
    mass_divisor = 100
    num_hybridization = len(Chem.HybridizationType.values)
    mininum_ring_size = 3
    maximum_ring_size = 8

    for atom in mol.GetAtoms():
        node_feature = []
        node_feature = node_feature + [
            1 if atom.GetAtomicNum() == i else 0
            for i in range(1, maximum_num_atoms + 1)
        ]
        node_feature.append(atom.GetDegree())
        node_feature.append(atom.GetExplicitValence())
        node_feature.append(atom.GetFormalCharge())
        node_feature = node_feature + [
            1 if atom.GetHybridization() == Chem.HybridizationType.values[i] else 0
            for i in range(num_hybridization)
        ]
        node_feature.append(atom.GetImplicitValence())
        node_feature.append(int(atom.GetIsAromatic()))
        node_feature.append(atom.GetMass() / mass_divisor)
        node_feature.append(int(atom.GetNoImplicit()))
        node_feature.append(atom.GetNumExplicitHs())
        node_feature.append(atom.GetNumImplicitHs())
        node_feature.append(atom.GetNumRadicalElectrons())
        node_feature.append(atom.GetTotalDegree())
        node_feature.append(atom.GetTotalNumHs())
        node_feature.append(atom.GetTotalValence())
        node_feature.append(int(atom.IsInRing()))
        node_feature = node_feature + [
            1 if atom.IsInRingSize(i) else 0
            for i in range(mininum_ring_size, maximum_ring_size + 1)
        ]
        node_features.append(node_feature)

    node_features = np.asarray(node_features)
    # return torch.tensor(node_features, dtype=torch.float)
    return node_features


def get_edge_features(mol: Chem.Mol) -> torch.Tensor:
    all_edge_feature = []
    for bond in mol.GetBonds():
        edge_feature = []
        edge_feature.append(bond.GetBondTypeAsDouble())
        edge_feature.append(bond.GetIsAromatic())
        edge_feature.append(bond.GetIsConjugated())
        edge_feature.append(
            list(Chem.BondStereo.values.keys())[
                list(Chem.BondStereo.values.values()).index(bond.GetStereo())
            ]
        )
        edge_feature.append(bond.IsInRing())
        all_edge_feature.append(edge_feature)
        all_edge_feature.append(edge_feature)

    all_edge_feature = np.asarray(all_edge_feature)
    # return torch.tensor(all_edge_feature, dtype=torch.float)
    return all_edge_feature


def get_edge_index(mol: Chem.Mol) -> torch.Tensor:
    """
    adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
    row, col = np.where(adj_matrix)
    coo = np.array(list(zip(row, col)))
    coo = np.reshape(coo, (2, -1))
    """
    coo = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        coo.append([start, end])
        coo.append([end, start])
    # return torch.tensor(coo, dtype=torch.long).T
    return np.asarray(coo).astype(int).T


def get_mol_graph_attr(mol: Chem.Mol, mol_graph: nx.Graph, key: str) -> torch.Tensor:
    feat = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        feat.append(float(mol_graph.nodes[idx][key]))
    return torch.tensor(feat, dtype=torch.float)


def process_to_pyg_data(data: Union[Chem.Mol, str], **kwargs) -> Data:
    if isinstance(data, Chem.Mol):
        mol = data
    elif isinstance(data, str):
        mol = get_mol_from_smiles(data, **kwargs)
    else:
        mol = Chem.MolFromSmiles("")
    assert mol is not None, "Failed to parse the input data"

    node_features = get_node_features(mol)
    edge_features = get_edge_features(mol)
    edge_index = get_edge_index(mol)
    return Data(x=node_features, edge_attr=edge_features, edge_index=edge_index)


def process_to_dict(data: Union[Chem.Mol, str]) -> dict:
    if isinstance(data, Chem.Mol):
        mol = data
    elif isinstance(data, str):
        mol = get_mol_from_smiles(data)
    else:
        mol = Chem.MolFromSmiles("")
    assert mol is not None, "Failed to parse the input data"

    node_features = get_node_features(mol)
    edge_features = get_edge_features(mol)
    edge_index = get_edge_index(mol)
    return {
        "x": node_features,
        "edge_attr": edge_features,
        "edge_index": edge_index,
    }


class SmilesFingerprint:
    def __init__(self, radius: int = 2, nBits: int = 2048, **kwargs):
        self.radius = radius
        self.nBits = nBits
        self.morgan_fp = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=nBits, **kwargs
        )

        self._data = {}

    @lru_cache(maxsize=128)
    def __call__(self, smiles: str) -> np.ndarray:
        if type(smiles) != str or smiles == "":
            return np.zeros(self.nBits, dtype=np.float32)
        if smiles not in self._data:
            try:
                mol = get_mol_from_smiles(smiles)
                mol = clean_up_mol(mol)
                fp = self.morgan_fp.GetFingerprint(mol)
                self._data[smiles] = np.array(fp, dtype=np.float32)
            except:
                self._data[smiles] = np.zeros(self.nBits, dtype=np.float32)
        return self._data[smiles]
