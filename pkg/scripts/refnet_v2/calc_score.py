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
from ...models.ref_net_v2 import (
    BidirectionalPipe,
    GraphAttnEncoder,
    PyGDataInputLayer,
    RefNetV2,
    RegressionHead,
)
from ...utils.mol_feat_v2 import get_edge_features, get_edge_index, get_node_features
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
    parser.add_argument(
        "--weight_path",
        type=str,
        default="/data03/gtguo/del/refnet_weight/ca9_zip_500/model.pt",
    )
    parser.add_argument("--save_path", type=str, default="./save.csv")
    parser.add_argument("--batches_per_save", type=int, default=100)
    # dataset
    # parser.add_argument(
    #     "--dataset_fpath",
    #     type=str,
    #     default="/data03/gtguo/data/chembl/lmdb/target_hits/ca9/ca9_active_thr6.0.lmdb",
    # )
    parser.add_argument(
        "--dataset_fpath",
        type=str,
        default="/data03/gtguo/data/chemdiv/lmdb/chemdiv.lmdb",
    )
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

        affinity_pred = net.get_affinity(molecule_node_vec, data.batch, softplus=False)[
            :, 0
        ]
        return affinity_pred

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

    print("Reading dataset")
    # get the suffix of the dataset
    suffix = os.path.splitext(args.dataset_fpath)[1]

    if suffix == ".lmdb":
        smiles_dataset = LMDBDataset.readonly_raw(*os.path.split(args.dataset_fpath))

    elif suffix == ".txt":

        def handle_txt(fpath):
            # * here we load all the data into memory
            with open(fpath, "r") as f:
                smiles = f.readlines()
            smiles = [s.strip() for s in smiles]
            smiles_dataset = [{"smiles": s} for s in smiles]

            # * check if the first line is the header
            is_header = False
            try:
                get_mol_from_smiles(smiles[0])
            except:
                is_header = True
            smiles_dataset = smiles_dataset[1:] if is_header else smiles_dataset

            # * here a static lmdb dataset is created in situ
            _fdir, _fname = os.path.split(fpath)
            _fname = os.path.splitext(_fname)[0]
            _fname = os.path.join(_fdir, f"{_fname}.lmdb")

            smiles_dataset_lmdb = LMDBDataset.static_from_others(
                dataset=smiles_dataset,
                processed_dir=_fdir,
                processed_fname=_fname,
                forced_process=True,
            )
            return smiles_dataset_lmdb

        fpath_wo_suffix = os.path.splitext(args.dataset_fpath)[0]

        if os.path.exists(fpath_wo_suffix + ".lmdb"):
            smiles_dataset = LMDBDataset.readonly_raw(
                *os.path.split(fpath_wo_suffix + ".lmdb")
            )
        else:
            smiles_dataset = handle_txt(args.dataset_fpath)

    else:
        raise NotImplementedError(f"Dataset format {suffix} not supported for now")

    inference_dataset = CollateDataset(smiles_dataset)
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # print the length of the dataset
    # logger.info(f"Dataset length: {len(inference_dataset)}")

    model.load_state_dict(torch.load(args.weight_path))

    model.eval()
    model.to(device)

    def save_result(save_path, affinity_preds, idx_now):
        idxs = (
            np.arange(len(affinity_preds))
            + (idx_now // args.batches_per_save)
            * args.batch_size
            * args.batches_per_save
        )
        smiles = [smiles_dataset[idx]["smiles"] for idx in idxs]
        with open(save_path, "w") as f:
            for s, p in zip(smiles, affinity_preds):
                f.write(f"{s},{p}\n")

    logger.info("Start inference")
    with torch.no_grad():
        affinity_preds = []
        for i, data in enumerate(tqdm(inference_loader)):
            data = move_data_to_device(data)
            affinity_pred = inference_forward(model, data)
            affinity_preds.append(affinity_pred.cpu().numpy())
            if (i + 1) % args.batches_per_save == 0:
                affinity_preds = np.concatenate(affinity_preds)
                save_result(args.save_path, affinity_preds, i)
                affinity_preds = []
        if len(affinity_preds) > 0:
            affinity_preds = np.concatenate(affinity_preds)
            save_result(args.save_path, affinity_preds, i)

    logger.info("Inference finished")

    # print the length of the file written
    # logger.info(f"File length: {len(open(args.save_path).readlines())}")
