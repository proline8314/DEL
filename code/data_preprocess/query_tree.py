import copy
import os
import pickle
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from utils import (add_atom_with_bonds, clean_up_mol, get_mol_from_smiles,
                   get_peripheral_atom_idxs, get_standardized_smiles,
                   remove_atoms)


class SubstructQuery:
    def __init__(self, query_repr: Union[str, Chem.Mol]) -> "SubstructQuery":
        self.query_repr = query_repr
        self.query_mol = self._get_mol(query_repr)

        self.query_tree = None
        self.current_node = 0

        self._get_init_qtree()

    @property
    def num_nodes(self) -> int:
        return len(self.query_tree.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.query_tree.edges)

    @property
    def num_qatoms(self) -> int:
        return self.query_mol.GetNumAtoms()

    # * Private methods

    def _get_mol(self, repr: Union[str, Chem.Mol]) -> Chem.Mol:
        if isinstance(repr, str):
            mol = get_mol_from_smiles(repr)
        elif isinstance(repr, Chem.Mol):
            mol = copy.deepcopy(repr)
        else:
            raise ValueError("Invalid input representation.")

        mol = clean_up_mol(mol)
        if mol is None:
            raise ValueError(
                "Failed to get the molecule from the input representation."
            )
        return mol

    def _get_init_qtree(self) -> None:
        """
        Initial query tree have only one branch,
        stringing from the root node(complete query molecule) to the last node(core molecule).
        """
        self.query_tree = nx.DiGraph()

        self.query_tree.add_node(0, content=self.query_mol)

        current_mol = copy.deepcopy(self.query_mol)

        while True:
            if current_mol.GetNumAtoms() == 0:
                raise ValueError("The query molecule is empty.")

            prph_atom_idxs = get_peripheral_atom_idxs(current_mol)
            prph_atom_num = len(prph_atom_idxs)

            if prph_atom_num == 0 and self._is_core(current_mol):
                break

            # if prph_atom_num == 0:
            #     raise ValueError("Failed to find the peripheral atom.")

            current_mol, contravariant_idx = remove_atoms(
                current_mol, prph_atom_idxs, return_contravariant_idxs=True
            )
            self._add_node_to_qtree(
                self.num_nodes,
                self.num_nodes - 1,
                content=current_mol,
                edge_score=prph_atom_num,
                contravariant_idx=contravariant_idx,
            )

    def _get_extended_mol(
        self,
        mol: Chem.Mol,
        complete_mol: Chem.Mol,
        mode: Literal["bfs", "bfs_complement", "single"] = "bfs",
        *,
        contravariant_idx: Optional[Sequence[int]] = None,
        complement_ratio: float = 0.5,
    ) -> Union[Chem.Mol, List[Chem.Mol]]:
        """
        Get the extended molecule based on the query molecule and the complete molecule.

        @param mol: The query molecule.
        @param complete_mol: The complete molecule.
        @param mode:
        - "bfs" mode will extend the molecule based on the BFS traversal of the query molecule w/ depth equals to 1.
        - "bfs_complement" mode will give out a randomly BFS-extended molecule and its complement based on the complement ratio.
        - "single" mode will give out a set of single-atom-extented molecules.
        """
        # get the matched atoms
        if contravariant_idx is None:
            contravariant_idx = complete_mol.substructMatch(mol)[0]

        # get neighbor atoms and bonds
        neighbor_atom_idxs = {idx: self._get_neighbor_atoms_idx(complete_mol, idx) for idx in contravariant_idx}

    def _get_parent_nodes(self, node: int) -> List[int]:
        return list(self.query_tree.predecessors(node))

    def _get_children_nodes(self, node: int) -> List[int]:
        return list(self.query_tree.successors(node))

    def _get_neighbor_atoms_idx(self, mol: Chem.Mol, atom_idx: int) -> List[int]:
        return [a.GetIdx() for a in mol.GetAtomWithIdx(atom_idx).GetNeighbors()]

    def _get_bond_type(self, mol: Chem.Mol, atom1_idx: int, atom2_idx: int) -> int:
        return mol.GetBondBetweenAtoms(atom1_idx, atom2_idx).GetBondTypeAsDouble()

    def _contravariant_to_covariant(self, contravariant_idx: Sequence[int]) -> Sequence[int]:
        """
        Convert the contravariant index to the covariant index.
        Cov[Idx] == Contra.IndexOf(Idx)
        Contra[Idx] == Cov.IndexOf(Idx)
        e.g. 
        contra_idx = [2, 3, 4, 5, 6]
        return [-1, -1, 0, 1, 2, 3, 4]
        """
        covariant_idx = [-1] * (max(contravariant_idx) + 1)
        for i, idx in enumerate(contravariant_idx):
            covariant_idx[idx] = i
        return covariant_idx

    def _add_node_to_qtree(
        self,
        node: int,
        parent: int,
        content: Any,
        edge_score: int,
        contravariant_idx: Sequence[int] = None,
    ) -> None:
        content = copy.deepcopy(content)
        self.query_tree.add_node(node, content=content, matched=None)
        self.query_tree.add_edge(
            parent, node, score=edge_score, contravariant_idx=contravariant_idx
        )

    def _is_single_atom(self, mol: Chem.Mol) -> bool:
        return mol.GetNumAtoms() == 1

    def _is_double_atom(self, mol: Chem.Mol) -> bool:
        return mol.GetNumAtoms() == 2 and mol.GetBondBetweenAtoms(0, 1) is not None

    def _is_linked_ring(self, mol: Chem.Mol) -> bool:
        mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
        return mol.GetRingInfo().NumRings() > 0

    def _is_ring(self, mol: Chem.Mol) -> bool:
        mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
        return mol.GetRingInfo().NumRings() > 0 and all(
            [a.IsInRing() for a in mol.GetAtoms()]
        )

    def _is_core(self, mol: Chem.Mol) -> bool:
        return (
            self._is_single_atom(mol)
            or self._is_double_atom(mol)
            or self._is_linked_ring(mol)
        )

    def _is_identical_mols(self, mol1: Chem.Mol, mol2: Chem.Mol) -> bool:
        return get_standardized_smiles(mol1) == get_standardized_smiles(mol2)

    def _update_matched_status(self, node: int, matched: bool) -> None:
        self.query_tree.nodes[node]["matched"] = matched

    def _flush_qtree(self) -> None:
        for node in self.query_tree.nodes:
            self._update_matched_status(node, None)

    # * Public methods

    def display_qtree(self) -> None:
        print("Query tree:")
        for node in self.query_tree.nodes:
            content = self.query_tree.nodes[node]["content"]
            print(f"Node {node}: {get_standardized_smiles(content)}")
        print("Edges:")
        for edge in self.query_tree.edges:
            score = self.query_tree.edges[edge]["score"]
            contravariant_idx = self.query_tree.edges[edge]["contravariant_idx"]
            print(
                f"Edge {edge[0]} -> {edge[1]}: Scored {score}, Contravariant index {contravariant_idx}"
            )


if __name__ == "__main__":
    query = "OC1=CC=C(C(C(O)=O)N)C=C1"
    # query = "O=C(NC(C(O)C(OC1CC(C(OC(C2=CC=CC=C2)=O)C(C(CO3)(OC(C)=O)C3CC4O)C4C(C5OC(C)=O)=O)(O)C(C)(C)C5=C1C)=O)C6=CC=CC=C6)C7=CC=CC=C7"
    print("Input query:", query)
    query = SubstructQuery(query)
    query.display_qtree()
