import copy
import os
import pickle
from typing import Any, Hashable
from warnings import warn

import lmdb
import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from tqdm import tqdm


def get_bb_dict(dir_path) -> list[dict]:
    env = lmdb.open(dir_path, subdir=False,
                    readonly=False,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=1, map_size=1099511627776)
    txn = env.begin(write=True)
    output_dict = pickle.loads(txn.get("0".encode()))
    return output_dict


def smiles_to_mol_graph(smiles: str, return_mol_instance: bool = False, keep_all_hydrogen: bool = True) -> nx.Graph | tuple[nx.Graph, Chem.Mol]:
    """
    Transform a SMILES string to a molecular graph
    """
    if smiles == "":
        print("Empty SMILES string")
        return nx.Graph()

    mol = Chem.MolFromSmiles(smiles)
    # remove salt
    remover = SaltRemover()
    mol = remover.StripMol(mol)

    mol_graph = nx.Graph()
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        # filter hydrogen
        if not keep_all_hydrogen and atom_symbol == "H":
            continue
        mol_graph.add_node(atom_idx, atom_symbol=atom_symbol)
    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        # filter hydrogen
        if not keep_all_hydrogen and (atom1_idx == "H" or atom2_idx == "H"):
            continue
        # TODO maybe useful to consider passing rdkit bondtype afterwards
        bond_type = str(bond.GetBondType())
        mol_graph.add_edge(atom1_idx, atom2_idx, bond_type=bond_type)
    if return_mol_instance:
        return mol_graph, mol
    else:
        return mol_graph


def reduce_graph_to_core(mol_graph: nx.Graph) -> nx.Graph:
    """
    Keep reducing the molecular graph, until:
    1. The graph contains only one atom
    2. The graph contains a diatomic chain
    3. The graph is made of rings and linkers
    """
    core_graph = copy.deepcopy(mol_graph)
    while True:
        # get the atom indices of the peripheral atoms
        peripheral_atom_indices = [atom_idx for atom_idx in core_graph.nodes if len(
            list(core_graph.neighbors(atom_idx))) == 1]
        # no peripheral atoms means the graph is made of rings and linkers between rings
        # ? or there is no atom
        if len(peripheral_atom_indices) == 0:
            return core_graph
        # single atom or diatomic chain
        elif len(peripheral_atom_indices) <= 2 and len(peripheral_atom_indices) == len(core_graph.nodes):
            return core_graph
        else:
            core_graph.remove_nodes_from(peripheral_atom_indices)


def expand_core_to_skeleton(
        bb_mol_graph: nx.Graph,
        core_of_bb_mol_graph: nx.Graph,
        template_mol_graph: nx.Graph
):
    def add_atom(mol_graph: nx.Graph, new_atom_idx: int, root_atom_idx: int, new_atom_symbol: str, bond_type: str) -> nx.Graph:
        mol_graph.add_node(new_atom_idx, atom_symbol=new_atom_symbol)
        mol_graph.add_edge(new_atom_idx, root_atom_idx, bond_type=bond_type)
        return mol_graph

    def r_branch_from_single_atom(
            atom_idx: int,
            current_graph: nx.Graph,
            bb_mol_graph: nx.Graph,
            template_mol_graph: nx.Graph,
            public_occupied_atom_indeces: list
    ):
        for neighbor in bb_mol_graph.neighbors(atom_idx):
            if neighbor in public_occupied_atom_indeces:
                continue
            branch_attempt_graph = copy.deepcopy(current_graph)
            branch_attempt_graph = add_atom(branch_attempt_graph, neighbor, atom_idx,
                                            bb_mol_graph.nodes[neighbor]["atom_symbol"], bb_mol_graph.edges[neighbor, atom_idx]["bond_type"])
            # check atom symbols only
            gm = nx.algorithms.isomorphism.GraphMatcher(
                template_mol_graph, branch_attempt_graph, node_match=lambda n1, n2: n1['atom_symbol'] == n2['atom_symbol'], edge_match=lambda e1, e2: e1['bond_type'] == e2['bond_type'])
            if gm.subgraph_is_isomorphic():
                public_occupied_atom_indeces.append(neighbor)
                current_graph = r_branch_from_single_atom(
                    neighbor, branch_attempt_graph, bb_mol_graph, template_mol_graph, public_occupied_atom_indeces)
            # else the current graph stays unchanged
        return current_graph

    public_occupied_atom_indeces = [
        atom_idx for atom_idx in core_of_bb_mol_graph.nodes]
    skeleton_of_bb_mol_graph = copy.deepcopy(core_of_bb_mol_graph)
    for core_atoms in core_of_bb_mol_graph.nodes:
        skeleton_of_bb_mol_graph = r_branch_from_single_atom(
            core_atoms, skeleton_of_bb_mol_graph, bb_mol_graph, template_mol_graph, public_occupied_atom_indeces)
    return skeleton_of_bb_mol_graph


def get_bb_skeleton(bb_mol_graph: nx.Graph, template_mol_graph: nx.Graph):
    # TODO: this method suits for reactions that do not form rings, need a template-based version soon, referring to J Chem Inf Model 2023, 63 (15), 4641-4653. SI.
    def expose_amide(mol_g: nx.Graph):
        g_copy = copy.deepcopy(mol_g)
        for atom_idx in mol_g.nodes:
            if mol_g.nodes[atom_idx]["atom_symbol"] != "N":
                continue
            for neighbor in mol_g.neighbors(atom_idx):
                if len([nn for nn in mol_g.neighbors(neighbor)]) > 1:
                    continue
                g_copy.remove_edge(neighbor, atom_idx)
                g_copy.remove_node(neighbor)
        return g_copy

    core_of_bb_mol_graph = reduce_graph_to_core(bb_mol_graph)
    skeleton_of_bb_mol_graph = expand_core_to_skeleton(
        bb_mol_graph=bb_mol_graph, core_of_bb_mol_graph=core_of_bb_mol_graph, template_mol_graph=template_mol_graph)
    skeleton_of_bb_mol_graph = expose_amide(skeleton_of_bb_mol_graph)
    return skeleton_of_bb_mol_graph


def match_motif_to_mol(mol: nx.Graph, bb_motif_dict: dict[str, nx.Graph]) -> nx.Graph:
    def node_match(n1, n2):
        return n1['atom_symbol'] == n2['atom_symbol'] and n1['motif'] is None

    def flexible_node_match(n1, n2):
        return n1['atom_symbol'] == n2['atom_symbol']

    def edge_match(e1, e2):
        return e1['bond_type'] == e2['bond_type']

    def r_find_nearest_motif_assigned_neighbor(mol: nx.Graph, node_idx: int, visited: list[int]):
        cmol = copy.deepcopy(mol)
        visited.append(node_idx)
        passed_motif_list = []
        # iterate over neighbors to get motif information
        for neighbor_idx in cmol.neighbors(node_idx):
            if cmol.nodes[neighbor_idx]["motif"] is not None:
                passed_motif = cmol.nodes[neighbor_idx]["motif"]
            # if the neighbor is not motif-assigned and not visited, recursively find the motif
            elif neighbor_idx not in visited:
                cmol, passed_motif = r_find_nearest_motif_assigned_neighbor(
                    cmol, neighbor_idx, visited)
            # prevent infinite loop
            else:
                continue
            passed_motif_list.append(passed_motif)
        passed_motif_list = np.unique(passed_motif_list)

        if len(passed_motif_list) > 1:
            motif_to_be_assigned = min(passed_motif_list)
            warn(f"Node {node_idx} has more than one motif-assigned neighbor, only the smallest one {motif_to_be_assigned} is considered")
        elif len(passed_motif_list) == 1:
            motif_to_be_assigned = passed_motif_list[0]
        else:
            # ?! a single unmatched ring could escape from the above logic
            # pick a big number (> num of bb) to avoid transferring to another atoms
            motif_to_be_assigned = 10
            warn(
                f"Node {node_idx} has escape the motif assignment and is assigned to a dummy motif {motif_to_be_assigned}")

        cmol.nodes[node_idx]["motif"] = motif_to_be_assigned
        return cmol, motif_to_be_assigned

    mol_copy = copy.deepcopy(mol)
    for node_idx in mol_copy.nodes:
        mol_copy.nodes[node_idx]["motif"] = None

    is_flexible_matching = False
    for motif_name, motif in bb_motif_dict.items():
        # ? special case for empty motif
        if len(motif.nodes) == 0:
            continue
        # ! A Band-aid solution to save flexible state for next subgraphs
        if not is_flexible_matching:
            gm = GraphMatcher(
                mol_copy, motif, node_match=node_match, edge_match=edge_match)
        else:
            gm = GraphMatcher(
                mol_copy, motif, node_match=flexible_node_match, edge_match=edge_match)
        if not gm.subgraph_is_isomorphic():
            # TODO push re-examining request
            warn(
                f"motif {motif_name} is not isomorphic to the molecule, due to probable wrong motif assignment")
            # ! switch to flexible matching
            motif.remove_nodes_from(
                [idx for idx in motif.nodes if len(list(motif.neighbors(idx))) == 1])
        for node_idx in gm.mapping.keys():
            mol_copy.nodes[node_idx]["motif"] = motif_name

        # ! flexible check, check if there are any another motif unmatched
        gm = GraphMatcher(
            mol_copy, motif, node_match=node_match, edge_match=edge_match)
        if not gm.subgraph_is_isomorphic():
            is_flexible_matching = False
            continue

        # ! if so, activate flexible matching for next bb
        is_flexible_matching = True
        warn(f"flexible matching activated because of bb-{motif_name}")
        # ! first match all possible atoms
        while True:
            for node_idx in gm.mapping.keys():
                mol_copy.nodes[node_idx]["motif"] = motif_name
            # ! new matcher assigned for the rest of the molecule
            gm = GraphMatcher(
                mol_copy, motif, node_match=node_match, edge_match=edge_match)
            if not gm.subgraph_is_isomorphic():
                break
            else:
                pass

    for node_idx in list(mol_copy.nodes):
        if mol_copy.nodes[node_idx]["motif"] is not None:
            continue
        mol_copy, _ = r_find_nearest_motif_assigned_neighbor(
            mol_copy, node_idx, [])
    return mol_copy


DNA_LINKER_MOTIF = "O=C(NC)C(C)N"   # suits for 002
GENERAL_LINKER = "O=C(NC)"


def add_node_attribute(mol: nx.Graph, node_attribute: Hashable, default: Any):
    cmol = copy.deepcopy(mol)
    for idx in cmol.nodes:
        cmol.nodes[idx][node_attribute] = default
    return cmol


def assign_genetal_linker(mol: nx.Graph):
    # * The methyl group attached to the amide as the start
    cmol = copy.deepcopy(mol)
    for idx in cmol.nodes:
        cmol.nodes[idx]["is_start"] = 0
    for idx in cmol.nodes:
        neighbor = list(cmol.neighbors(idx))
        is_start = cmol.nodes[idx]["atom_symbol"] == "C" and len(
            neighbor) == 1 and cmol.nodes[neighbor[0]]["atom_symbol"] == "N"
        if is_start:
            cmol.nodes[idx]["is_start"] = 1
            break
        else:
            pass
    return cmol


def assign_mol(mol: nx.Graph, assigned_template_mol: nx.Graph, return_idx: bool = False):
    cmol = copy.deepcopy(mol)
    for idx in cmol.nodes:
        cmol.nodes[idx]["is_start"] = 0
    gm = GraphMatcher(cmol, assigned_template_mol, node_match=lambda n1,
                      n2: n1["atom_symbol"] == n2["atom_symbol"])
    if not gm.subgraph_is_isomorphic():
        raise ValueError("Unmatched motif")
    start_idx = -1
    for cmol_idx, tmpl_idx in gm.mapping.items():
        if assigned_template_mol.nodes[tmpl_idx]["is_start"] == 1:
            cmol.nodes[cmol_idx]["is_start"] = 1
            start_idx = cmol_idx
            break
        else:
            continue
    if return_idx:
        return cmol, start_idx
    else:
        return cmol


def assign_motif(linker_motif: str = DNA_LINKER_MOTIF, general_linker: str = GENERAL_LINKER, node_attribute: str = "topo_dist"):
    general_linker_mol = smiles_to_mol_graph(
        general_linker, keep_all_hydrogen=False)
    general_linker_mol = assign_genetal_linker(general_linker_mol)
    linker_mol = smiles_to_mol_graph(linker_motif, keep_all_hydrogen=False)
    linker_mol = assign_mol(linker_mol, general_linker_mol)
    return linker_mol


def assign_topo_dist_to_DNA(mol: nx.Graph, linker_mol: nx.Graph, node_attribute: str = "topo_dist"):

    def expand_topo_dist(mol: nx.Graph, idx: int, node_attribute: str = "topo_dist"):
        # * A function-defined node attribute name would make the code more readable (functional encoding)
        # * use bfs here
        # TODO rip out other node attributes as arguments, as well as visit record list

        cmol = copy.deepcopy(mol)
        current_dist = 0
        current_depth_nodes = [idx]
        next_depth_nodes = []
        while len(current_depth_nodes) > 0:
            for node in current_depth_nodes:
                cmol.nodes[node][node_attribute] = current_dist
                for neighbor in cmol.neighbors(node):
                    if cmol.nodes[neighbor][node_attribute] is not None:
                        continue
                    next_depth_nodes.append(neighbor)
            current_depth_nodes = next_depth_nodes
            next_depth_nodes = []
            current_dist += 1
        return cmol

    mol, start_idx = assign_mol(mol, linker_mol, return_idx=True)
    mol = add_node_attribute(mol, node_attribute, None)
    mol = expand_topo_dist(mol, start_idx, node_attribute=node_attribute)
    return mol


if __name__ == "__main__":
    dataset_path = "/data02/gtguo/DEL/data/dataset"
    dataset_name = "acs.jcim.2c01608"

    dataset_fpath = os.path.join(dataset_path, dataset_name, "counts.lmdb")

    bb_smiles_dict = get_bb_dict(
        os.path.join(dataset_path, dataset_name, "bbsmiles.lmdb"))
    bb_skl_graph_dict = [{}, {}, {}]

    linker_mol = assign_motif()

    env = lmdb.open(dataset_fpath, subdir=False,
                    readonly=False,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=1, map_size=1099511627776)
    txn = env.begin()
    feat_env = lmdb.open(os.path.join(
        dataset_path, dataset_name, "features.lmdb"), subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1, map_size=1099511627776)
    feat_txn = feat_env.begin(write=True)

    for key, value in txn.cursor():
        data_dict = pickle.loads(value)
        mol_graph, mol = smiles_to_mol_graph(
            data_dict["mol_smiles"], return_mol_instance=True)
        bb_idxs = [data_dict["BB1_id"],
                   data_dict["BB2_id"], data_dict["BB3_id"]]
        bb_motif_dict = {}
        for i, bb_idx in enumerate(bb_idxs):
            if bb_idx not in bb_skl_graph_dict[i].keys():
                bb_smiles = bb_smiles_dict[i][bb_idx]
                bb_smiles = bb_smiles if type(bb_smiles) is not float else ""
                bb_skl_graph_dict[i][bb_idx] = get_bb_skeleton(
                    smiles_to_mol_graph(bb_smiles), mol_graph)
            else:
                pass
            bb_motif_dict[f"{i+1}"] = bb_skl_graph_dict[i][bb_idx]

        print("handling {}".format(data_dict["mol_id"]))
        mol_graph = match_motif_to_mol(mol_graph, bb_motif_dict)
        mol_graph = assign_topo_dist_to_DNA(mol_graph, linker_mol)

        data_dict["mol"] = mol
        data_dict["mol_graph"] = mol_graph
        feat_txn.put(key, pickle.dumps(data_dict, protocol=-1))

    txn.commit()
    env.close()
    feat_txn.commit()
    feat_env.close()
