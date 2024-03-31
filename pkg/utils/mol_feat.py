from typing import List, Optional, Tuple

import rdkit
import torch
from rdkit import Chem
from rdkit.Chem import rdmolops


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
    return torch.tensor(node_features, dtype=torch.float)


    