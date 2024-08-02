import copy
import io
import os
import pickle
from collections import defaultdict
from itertools import permutations
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
from warnings import warn

import networkx as nx
import numpy as np
import rdkit
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import IPythonConsole
from tqdm import tqdm
from utils import (clean_up_mol, find_mutual_substruct, get_mol_from_smiles,
                   remove_atoms)


def get_synthon_index(
    mol: Chem.Mol,
    syn_mol_dict: Dict[int, Chem.Mol],
    check_single_connected: bool = True,
):
    # Find the mutual substructure between the molecule and each synthon
    mcs_dict = {
        idx: find_mutual_substruct(mol, syn_mol)
        for idx, syn_mol in syn_mol_dict.items()
    }
    matches_dict = {idx: mol.GetSubstructMatches(mcs) for idx, mcs in mcs_dict.items()}
    # Search for the best combination of matches that maximizes the coverage of the molecule
    best_combination, syns_idx_sorted = search_best_combanation(
        matches_dict, mol.GetNumAtoms()
    )
    if len(get_idx_union(best_combination)) < mol.GetNumAtoms():
        warn(
            f"No combination found that covers all atoms for {Chem.MolToSmiles(mol)}, synthons: {[Chem.MolToSmiles(syn_mol_dict[idx]) for idx in syns_idx_sorted]}"
        )

    syn_idx_array = None
    if check_single_connected:
        syn_idx_array = assign_single_connected(
            mol, best_combination, syns_idx_sorted, syn_mol_dict
        )
    if syn_idx_array is not None:
        return syn_idx_array

    syn_idx_array = -np.ones(mol.GetNumAtoms(), dtype=int)
    for ns, idxs in enumerate(best_combination):
        idxs = np.array(idxs)
        syn_idx_array[idxs] = syns_idx_sorted[ns]
    return syn_idx_array


def assign_single_connected(mol, best_combination, syns_idx_sorted, syn_mol_dict):
    num_syns_idx = len(syns_idx_sorted)
    idx_perms = list(permutations(range(num_syns_idx)))
    # idx_perms = sorted(idx_perms, reverse=True)
    # check for topology of the DEL molecules
    ns_idx_list = [(ns, idx) for ns, idx in enumerate(best_combination)]

    for idx_perm in idx_perms:
        syn_idx_array = -np.ones(mol.GetNumAtoms(), dtype=int)
        # permute the synthon index
        ns_idx_list_perm = [ns_idx_list[idx] for idx in idx_perm]
        for ns, idxs in ns_idx_list_perm:
            idxs = np.array(idxs)
            syn_idx_array[idxs] = syns_idx_sorted[ns]
        if is_single_connected(
            mol, syn_idx_array, idxs_to_check=syns_idx_sorted, syn_mol_dict=syn_mol_dict
        ):
            return syn_idx_array

    warn(
        f"No single connected assignment found or ring breaking ocurrs for {Chem.MolToSmiles(mol)}"
    )
    return None


def is_single_connected(mol, syn_idx_array, idxs_to_check, syn_mol_dict) -> bool:
    is_single_connected = True
    mol = copy.deepcopy(mol)
    mol_idxs = np.array([a.GetIdx() for a in mol.GetAtoms()])

    for idx in idxs_to_check:
        atoms_to_remove = [int(i) for i in mol_idxs[syn_idx_array != idx]]
        mol_ = remove_atoms(mol, atoms_to_remove)
        syn_mol = syn_mol_dict[idx]
        if len(Chem.MolToSmiles(mol_).split(".")) > 1 or not has_same_ring_number(
            mol_, syn_mol
        ):
            is_single_connected = False
            break
    return is_single_connected


def has_same_ring_number(mol1, mol2):
    mol1.UpdatePropertyCache(strict=False)
    mol2.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol1)
    Chem.GetSymmSSSR(mol2)
    return mol1.GetRingInfo().NumRings() == mol2.GetRingInfo().NumRings()


def search_best_combanation(
    matches_dict: Dict[int, List[Tuple[int]]], num_tot_atoms: int
) -> Tuple[List[Tuple[int]], List[int]]:
    """
    Pick one from each synthon and combine them to maximize the coverage of the molecule
    Brute force search
    """
    num_syns = len(matches_dict)
    syns_idx_sorted = np.array(sorted(matches_dict.keys()))
    num_matches = np.array([len(matches_dict[idx]) for idx in syns_idx_sorted])

    zero_matches = num_matches == 0
    num_matches[num_matches == 0] = 1
    cumprod_matches = np.cumprod(num_matches)
    num_combinations = cumprod_matches[-1]

    def get_combination(idxs: List[int]):
        return [
            matches_dict[syns_idx_sorted[ns]][idxs[ns]]
            for ns in range(num_syns)
            if zero_matches[ns] == False
        ]

    def decode_compound_idx(compound_idx: int):
        idxs = [
            compound_idx // cumprod_matches[j] % num_matches[j] for j in range(num_syns)
        ]
        return idxs

    best_coverage = 0
    best_idxs = None

    for i in range(num_combinations):
        idxs = decode_compound_idx(i)
        coverage = get_idx_union(get_combination(idxs))
        if len(coverage) == num_tot_atoms:
            return get_combination(idxs), syns_idx_sorted[zero_matches == False]
        elif len(coverage) > best_coverage:
            best_coverage = len(coverage)
            best_idxs = idxs

    return get_combination(best_idxs), syns_idx_sorted[zero_matches == False]


def get_idx_union(idx_sets: Sequence[Sequence[int]]):
    idx_union = set()
    for idx_set in idx_sets:
        idx_union.update(set(idx_set))
    return idx_union


def get_synthon_index_from_smiles(
    smiles: str,
    syn_smiles_dict: Dict[int, str],
):
    mol = get_mol_from_smiles(smiles, sanitize=False)
    mol = clean_up_mol(mol)
    syn_mol_dict = {
        idx: get_mol_from_smiles(syn_smiles)
        for idx, syn_smiles in syn_smiles_dict.items()
    }
    syn_mol_dict = {idx: clean_up_mol(syn_mol) for idx, syn_mol in syn_mol_dict.items()}
    return mol, syn_mol_dict, get_synthon_index(mol, syn_mol_dict)


def display_mol_with_colors(
    mol,
    atom_idxs,
    fname=None,
    colors=[
        (1.0, 0.0, 0.0, 0.2),
        (0.0, 1.0, 0.0, 0.2),
        (0.0, 0.0, 1.0, 0.2),
        (1.0, 1.0, 0.0, 0.2),
        (0.0, 1.0, 1.0, 0.2),
        (1.0, 0.0, 1.0, 0.2),
        (0.0, 0.0, 0.0, 0.1),
    ],
):
    IPythonConsole.ipython_useSVG = True
    athighlight = defaultdict(list)
    arads = {}
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        syn_idx = atom_idxs[idx]
        athighlight[idx].append(colors[syn_idx])
        arads[idx] = 0.3

    bndhighlight = defaultdict(list)
    for b in mol.GetBonds():
        bgn = b.GetBeginAtomIdx()
        end = b.GetEndAtomIdx()
        bgn_syn_idx = atom_idxs[bgn]
        end_syn_idx = atom_idxs[end]
        if bgn_syn_idx == end_syn_idx:
            bid = b.GetIdx()
            bndhighlight[bid].append(colors[bgn_syn_idx])

    fname = "temp.png" if fname is None else fname
    d2d = Draw.rdMolDraw2D.MolDraw2DCairo(350, 440)
    d2d.DrawMoleculeWithHighlights(
        mol, "", dict(athighlight), dict(bndhighlight), arads, {}
    )
    d2d.FinishDrawing()
    d2d.WriteDrawingText(fname)
    bio = io.BytesIO(d2d.GetDrawingText())
    img = Image.open(bio)
    return img


if __name__ == "__main__":
    syn_smiles_dict = {
        0: "CC(=O)NC(CCC(N)=O)C(=O)O",
        1: "CC(=O)NC(CCC(N)=O)C(N)=O",
        2: "CNC(C1Cc(c(CN1)[nH]2)c3c2cccc3)=O",
    }
    smiles = "NC(=O)CC[C@@H](NC(=O)[C@@H](CCC(N)=O)NC(=O)[C@@H]1Cc2c([nH]c3ccccc23)CN1)C(=O)O"
    mol, syn_mol_dict, synthon_index = get_synthon_index_from_smiles(
        smiles, syn_smiles_dict
    )
    display_mol_with_colors(mol, synthon_index)
