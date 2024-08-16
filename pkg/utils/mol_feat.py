from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import rdkit
import torch
from rdkit import Chem
from rdkit.Chem import rdchem, rdmolops
from rdkit.Chem.SaltRemover import SaltRemover
from torch_geometric.data import Data


def get_node_features(mol: Chem.Mol, *, atom_type_list: Optional[List["str"]] = None) -> torch.Tensor:
    r"""According to Neural Message Passing for Quantum Chemistry, the node features are:
    - Atomic type: one-hot 
    - Atomic number: integer
    - Degree: integer # ? suspicous though for GNN
    - Aromatic: boolean
    - Hybridization: one-hot among SP, SP2, SP3 and others
    - Num Hs: integer
    - Formal charge: integer
    - Whether the atom is in the ring: boolean"""
    def one_hot_encode(value, value_list, include_none=False):
        return [1 if value == v else 0 for v in value_list] + ([int(value in value_list)] if include_none else [])
        
    node_features = []
    atom_type_list = atom_type_list or ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    hybridization_list = [rdkit.Chem.rdchem.HybridizationType.SP, rdkit.Chem.rdchem.HybridizationType.SP2, rdkit.Chem.rdchem.HybridizationType.SP3]
    for atom in mol.GetAtoms():
        node_feature = []
        node_feature = node_feature + one_hot_encode(atom.GetSymbol(), atom_type_list)
        node_feature.append(atom.GetAtomicNum())
        node_feature.append(atom.GetDegree())
        node_feature.append(int(atom.GetIsAromatic()))
        node_feature = node_feature + one_hot_encode(atom.GetHybridization(), hybridization_list, include_none=True)
        node_feature.append(atom.GetTotalNumHs())
        node_feature.append(atom.GetFormalCharge())
        node_feature.append(int(atom.IsInRing()))
        node_features.append(node_feature)
    node_features = np.asarray(node_features)
    return torch.tensor(node_features, dtype=torch.float)

def get_edge_features(mol: Chem.Mol) -> torch.Tensor:
    all_edge_feature = []
    for bond in mol.GetBonds():
        edge_feature = []
        edge_feature.append(bond.GetBondTypeAsDouble())
        """
        edge_feature.append(rdchem.BondStereo.values.index(bond.GetStereo()))
        edge_feature.append(bond.GetIsConjugated())
        """
        edge_feature.append(bond.IsInRing())
        all_edge_feature.append(edge_feature)
        all_edge_feature.append(edge_feature)

    all_edge_feature = np.asarray(all_edge_feature)
    return torch.tensor(all_edge_feature, dtype=torch.float)

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
    return torch.tensor(coo, dtype=torch.long).T

def get_mol_graph_attr(mol: Chem.Mol, mol_graph: nx.Graph, key: str) -> torch.Tensor:
    feat = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        feat.append(float(mol_graph.nodes[idx][key]))
    return torch.tensor(feat, dtype=torch.float)

def process_to_pyg_data(data: Union[Chem.Mol, str], *, atom_type_list: Optional[List[str]] = None) -> Data:
    if isinstance(data, Chem.Mol):
        mol = data
    elif isinstance(data, str):
        mol = Chem.MolFromSmiles(data)
    else:
        mol = Chem.MolFromSmiles("")
    assert mol is not None, "Failed to parse the input data"
    mol = SaltRemover().StripMol(mol)
    node_features = get_node_features(mol, atom_type_list=atom_type_list)
    edge_features = get_edge_features(mol)
    edge_index = get_edge_index(mol)
    return Data(x=node_features, edge_attr=edge_features, edge_index=edge_index)
    