import copy
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover, rdFMCS
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
USE_SR = False
LOGGER = logging.getLogger(__name__)
if USE_SR:
    SR = SaltRemover.SaltRemover()
UC = rdMolStandardize.Uncharger()
TE = rdMolStandardize.TautomerEnumerator()


def clean_up_mol(mol: Chem.Mol) -> Chem.Mol:
    """
    Clean up molecule.
    """
    mol_str = Chem.MolToSmiles(mol)
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
    except:
        LOGGER.error(f"Failed to clean up molecule: {mol_str}")
        raise ValueError(f"Failed to clean up molecule: {mol_str}")
    try:
        if USE_SR:
            mol = SR.StripMol(mol)
    except:
        LOGGER.info(f"Failed to strip molecule: {mol_str}")
    try:
        mol = UC.uncharge(mol)
    except:
        LOGGER.info(f"Failed to uncharge molecule: {mol_str}")
    try:
        mol = TE.Canonicalize(mol)
    except:
        LOGGER.info(f"Failed to canonicalize molecule: {mol_str}")
    return mol


def get_mol_from_smiles(smiles: str, **kwargs) -> Chem.Mol:
    """
    Get rdkit mol object from smiles string.
    """
    mol = Chem.MolFromSmiles(smiles, **kwargs)
    if mol is None:
        LOGGER.error(f"Failed to get mol from smiles: {smiles}")
        raise ValueError(f"Failed to get mol from smiles: {smiles}")
    return mol


def get_peripheral_atom_idxs(mol: Chem.Mol) -> List[int]:
    """
    Get peripheral atom indices.
    """
    peripheral_atom_idxs = []
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 1:
            peripheral_atom_idxs.append(atom.GetIdx())
    return peripheral_atom_idxs


def standardize_smiles(smiles: str, stereo: bool = False) -> Optional[str]:
    """
    Get standardized smiles string.
    """
    mol = get_mol_from_smiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    if not stereo:
        Chem.RemoveStereochemistry(mol)
    mol = clean_up_mol(mol)
    return Chem.MolToSmiles(mol)


def get_standardized_smiles(mol: Chem.Mol, stereo: bool = False) -> Optional[str]:
    """
    Get standardized smiles string from rdkit mol object.
    """
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    if not stereo:
        Chem.RemoveStereochemistry(mol)
    mol = clean_up_mol(mol)
    return Chem.MolToSmiles(mol)


def remove_atoms(
    mol: Chem.Mol, atom_idxs: List[int], return_contravariant_idxs: bool = False
) -> Union[Chem.Mol, Tuple[Chem.Mol, List[Tuple[int]]]]:
    """
    Remove atoms from molecule.
    """
    mol = copy.deepcopy(mol)
    emol = Chem.EditableMol(mol)
    # * Indexes are reassigned after removing atoms, so remove atoms in reverse order
    for atom_idx in sorted(atom_idxs, reverse=True):
        emol.RemoveAtom(atom_idx)

    mol_removed = emol.GetMol()
    if not return_contravariant_idxs:
        return mol_removed

    match_idxs = [
        mth
        for mth in mol.GetSubstructMatches(mol_removed)
        if set(mth) & set(atom_idxs) == set()
    ]
    return mol_removed, match_idxs


bond_type_like = Union[Chem.BondType, str, int]


def get_bond_type(bond_type: bond_type_like) -> Chem.BondType:
    """
    Get bond type from bond type like.
    """
    if isinstance(bond_type, Chem.BondType):
        return bond_type
    if isinstance(bond_type, str):
        return getattr(Chem.rdchem.BondType, bond_type)
    if isinstance(bond_type, int):
        return Chem.rdchem.BondType.values[bond_type]
    raise ValueError(f"Invalid bond type: {bond_type}")


def add_atom(mol: Chem.Mol, atomic_num: int, return_idx: bool = False) -> Chem.Mol:
    """
    Add atom to molecule.
    """
    mol = copy.deepcopy(mol)
    emol = Chem.EditableMol(mol)
    atom = Chem.Atom(atomic_num)
    atom_idx = emol.AddAtom(atom)
    if return_idx:
        return emol.GetMol(), atom_idx
    return emol.GetMol()


def add_bond(
    mol: Chem.Mol, atom_idx1: int, atom_idx2: int, bond_type: bond_type_like
) -> Chem.Mol:
    """
    Add bond to molecule.
    """
    mol = copy.deepcopy(mol)
    emol = Chem.EditableMol(mol)
    emol.AddBond(atom_idx1, atom_idx2, get_bond_type(bond_type))
    return emol.GetMol()


def add_atom_with_bonds(
    mol: Chem.Mol,
    atomic_num: int,
    bond_to_idxs: Union[int, Sequence[int]],
    bond_type: Union[bond_type_like, Sequence[bond_type_like]],
) -> Chem.Mol:
    """
    Add atom with bonds to molecule.
    """
    mol = copy.deepcopy(mol)
    emol = Chem.EditableMol(mol)
    atom = Chem.Atom(atomic_num)
    atom_idx = emol.AddAtom(atom)
    if isinstance(bond_to_idxs, int):
        bond_to_idxs = [bond_to_idxs]
        bond_type = [bond_type]
    assert len(bond_to_idxs) == len(bond_type)
    bond_type = [get_bond_type(bond_type_) for bond_type_ in bond_type]
    for bond_to_idx, bond_type_ in zip(bond_to_idxs, bond_type):
        emol.AddBond(atom_idx, bond_to_idx, bond_type_)
    return emol.GetMol()


def find_mutual_substruct(mol1: Chem.Mol, mol2: Chem.Mol, **kwargs) -> Chem.Mol:
    mcs = rdFMCS.FindMCS([mol1, mol2], **kwargs)
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    return mcs_mol


def find_different_substruct(mol1: Chem.Mol, mol2: Chem.Mol, **kwargs) -> Chem.Mol:
    mcs = rdFMCS.FindMCS([mol1, mol2], **kwargs)
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    diff_1 = Chem.DeleteSubstructs(mol1, mcs_mol)
    diff_2 = Chem.DeleteSubstructs(mol2, mcs_mol)
    return diff_1, diff_2
