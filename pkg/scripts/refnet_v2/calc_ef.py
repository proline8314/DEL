import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ...datasets.lmdb_dataset import LMDBDataset
from ...models.ref_net_v2 import (BidirectionalPipe, GraphAttnEncoder,
                                  PyGDataInputLayer, RefNetV2, RegressionHead)
from ...utils.mol_feat_v2 import (get_edge_features, get_edge_index,
                                  get_node_features)
from ...utils.utils import get_mol_from_smiles

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up arguments
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--weight_path", type=str, default="/data03/gtguo/del/refnet_weight/ca9_zip_500/model.pt")
    # dataset
    parser.add_argument("--positive_dataset_fpath", type=str, default="/data03/gtguo/data/chembl/lmdb/target_hits/ca9/ca9_active_thr6.0.lmdb")
    # parser.add_argument("--negative_dataset_fpath", type=str, default="/data03/gtguo/data/chembl/lmdb/target_hits/ca9/ca9_inactive_thr6.0.lmdb")
    parser.add_argument("--negative_dataset_fpath", type=str, default="/data03/gtguo/data/chemdiv/lmdb/chemdiv.lmdb")

    # dataloader
    parser.add_argument("--num_workers", type=int, default=16)

    # model
    parser.add_argument("--synthon_node_feat_dim", type=int, default=2048)
    parser.add_argument("--synthon_node_emb_dim", type=int, default=64)
    parser.add_argument(
        "--synthon_node_emb_method", type=str, default="Embedding-Linear"
    )
    parser.add_argument("--synthon_token_size", type=int, default=16)
    parser.add_argument("--synthon_edge_emb_dim", type=int, default=64)
    parser.add_argument("--mol_node_feat_dim", type=int, default=147)
    parser.add_argument("--mol_node_emb_dim", type=int, default=64)
    parser.add_argument("--mol_node_emb_method", type=str, default="Linear")
    parser.add_argument("--mol_edge_feat_dim", type=int, default=5)
    parser.add_argument("--mol_edge_emb_dim", type=int, default=64)
    parser.add_argument("--mol_edge_emb_method", type=str, default="Linear")
    parser.add_argument("--encoder_layers", type=int, default=5)

    args = parser.parse_args()

    # print arguments to log
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Set up random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Model
    synthon_handler = PyGDataInputLayer(
        args.synthon_node_feat_dim,
        args.synthon_node_emb_dim,
        args.synthon_node_emb_method,
        None,
        args.synthon_edge_emb_dim,
        "None",
        token_size=args.synthon_token_size,
    ).to(device)
    mol_handler = PyGDataInputLayer(
        args.mol_node_feat_dim,
        args.mol_node_emb_dim,
        args.mol_node_emb_method,
        args.mol_edge_feat_dim,
        args.mol_edge_emb_dim,
        args.mol_edge_emb_method,
    ).to(device)
    synthon_encoder = GraphAttnEncoder(
        args.synthon_node_emb_dim,
        args.synthon_edge_emb_dim,
        num_layers=args.encoder_layers,
    ).to(device)
    mol_encoder = GraphAttnEncoder(
        args.mol_node_emb_dim, args.mol_edge_emb_dim, num_layers=args.encoder_layers
    ).to(device)
    pipes = [
        BidirectionalPipe(args.synthon_node_emb_dim, args.mol_node_emb_dim).to(device)
        for _ in range(args.encoder_layers)
    ]
    yield_head = RegressionHead(
        args.synthon_edge_emb_dim, args.mol_edge_emb_dim, 1, "sigmoid"
    ).to(device)
    affinity_head = RegressionHead(
        args.mol_node_emb_dim, args.mol_node_emb_dim, 2, None
    ).to(device)

    model = RefNetV2(
        synthon_feat_input=synthon_handler,
        molecule_feat_input=mol_handler,
        synthon_encoder=synthon_encoder,
        molecule_encoder=mol_encoder,
        bidirectional_pipes=pipes,
        reaction_yield_head=yield_head,
        affinity_head=affinity_head,
    ).to(device)

    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")    

    # Train

    def move_data_to_device(data):
        for k, v in data.items():
            if isinstance(v, dict):
                move_data_to_device(v)
            else:
                data[k] = v.to(device)
        return data

    def inference_forward(net: RefNetV2, data):
        molecule_node_vec, molecule_edge_idx, molecule_edge_vec = (
            net.molecule_feat_input(data)
        )
        for i in range(net.n_layers):
            molecule_node_vec, molecule_edge_vec = net.molecule_encoder.layers[i](
                molecule_node_vec, molecule_edge_idx, molecule_edge_vec
            )
            molecule_node_vec[molecule_node_vec.isnan()] = 0.0

        affinity_pred = net.get_affinity(molecule_node_vec, data.batch)
        return affinity_pred
    
    def enrichment_factor(scores, is_active, percentage=0.1):
        # descending order
        sorted_idx = np.argsort(scores)[::-1]
        n_active = int(np.sum(is_active))
        n_total = len(scores)
        n_top = int(n_total * percentage)
        n_top_active = np.sum(is_active[sorted_idx[:n_top]])
        return n_top_active / n_active / percentage

    print("Reading dataset")
    # * pass feat_fn to LMDBDataset
    positive_dataset = LMDBDataset.readonly_raw(
        *os.path.split(args.positive_dataset_fpath)
    )
    negative_dataset = LMDBDataset.readonly_raw(
        *os.path.split(args.negative_dataset_fpath)
    )

    class CollateDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            data = self.dataset[idx]
            data = self.handle_data(data)
            return data
            
        def handle_data(self, data):
            smiles = data["smiles"]
            try:
                mol = get_mol_from_smiles(smiles)
            except:
                mol = get_mol_from_smiles(smiles, sanitize=False)

            x = get_node_features(mol)
            edge_index = get_edge_index(mol)
            edge_attr = get_edge_features(mol)

            x = torch.tensor(x, dtype=torch.float32)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            return data
        
    positive_dataset = CollateDataset(positive_dataset)
    negative_dataset = CollateDataset(negative_dataset)

    positive_loader = DataLoader(positive_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    negative_loader = DataLoader(negative_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model.load_state_dict(torch.load(args.weight_path))

    model.eval()
    model.to(device)

    logger.info("Start inference")
    y_true = []
    y_pred = []

    for data in tqdm(positive_loader):
        data = move_data_to_device(data)
        with torch.no_grad():
            y_pred.append(inference_forward(model, data).cpu().numpy())

    for data in tqdm(negative_loader):
        data = move_data_to_device(data)
        with torch.no_grad():
            y_pred.append(inference_forward(model, data).cpu().numpy())

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate([np.ones(len(positive_dataset)), np.zeros(len(negative_dataset))])
    
    def stats(y_pred, y_true):
        print(f"EF 0.1%: {enrichment_factor(y_pred, y_true, 0.001)}")
        print(f"EF 1%: {enrichment_factor(y_pred, y_true, 0.01)}")
        print(f"EF 2%: {enrichment_factor(y_pred, y_true, 0.02)}")
        print(f"EF 5%: {enrichment_factor(y_pred, y_true, 0.05)}")
        print(f"EF 10%: {enrichment_factor(y_pred, y_true, 0.1)}")

        print(f"ROC-AUC: {roc_auc_score(y_true, y_pred)}")

        print(f"Mean positive score: {np.mean(y_pred[y_true == 1])}")
        print(f"Mean negative score: {np.mean(y_pred[y_true == 0])}")
    

    print("target lambda")
    stats(y_pred[:, 0], y_true)

    print("matrix lambda")
    stats(y_pred[:, 1], y_true)

    print("negative matrix lambda")
    stats(-y_pred[:, 1], y_true)