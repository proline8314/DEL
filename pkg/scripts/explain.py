import argparse
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorboard as tb
import torch
from datasets.graph_dataset import GraphDataset
from models.ref_net import DELRefDecoder, DELRefEncoder, DELRefNet
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader
from tqdm import tqdm

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up arguments
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--name", type=str, default="hrp_large_corr_400")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--load_path", type=str, default="/data02/gtguo/DEL/data/weights/refnet/")
    # dataset
    parser.add_argument("--target_name", type=str, default="hrp")
    parser.add_argument("--fp_size", type=int, default=2048)

    # model encoder
    parser.add_argument("--enc_node_feat_dim", type=int, default=19)
    parser.add_argument("--enc_edge_feat_dim", type=int, default=2)
    parser.add_argument("--enc_node_embedding_size", type=int, default=64)
    parser.add_argument("--enc_edge_embedding_size", type=int, default=64)
    parser.add_argument("--enc_n_layers", type=int, default=5)
    parser.add_argument("--enc_gat_n_heads", type=int, default=4)
    parser.add_argument("--enc_gat_ffn_ratio", type=int, default=4)
    parser.add_argument("--enc_with_fp", action="store_true")
    parser.add_argument("--enc_fp_embedding_size", type=int, default=32)
    parser.add_argument("--enc_fp_ffn_size", type=int, default=128)
    parser.add_argument("--enc_fp_gated", action="store_true")
    parser.add_argument("--enc_fp_n_heads", type=int, default=4)
    parser.add_argument("--enc_fp_size", type=int, default=256)
    parser.add_argument("--enc_fp_to_gat_feedback", type=str, default="add")
    parser.add_argument("--enc_gat_to_fp_pooling", type=str, default="mean")

    # model decoder
    parser.add_argument("--dec_node_input_size", type=int, default=64)
    parser.add_argument("--dec_node_emb_size", type=int, default=64)
    parser.add_argument("--dec_fp_input_size", type=int, default=32)
    parser.add_argument("--dec_fp_emb_size", type=int, default=64)
    parser.add_argument("--dec_output_size", type=int, default=2)
    parser.add_argument("--dec_with_fp", action="store_true")
    parser.add_argument("--dec_with_dist", action="store_true")

    # loss
    parser.add_argument("--target_size", type=int, default=4)
    parser.add_argument("--label_size", type=int, default=6)
    parser.add_argument("--matrix_size", type=int, default=2)

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
    encoder = DELRefEncoder(
        node_feat_dim=args.enc_node_feat_dim,
        edge_feat_dim=args.enc_edge_feat_dim,
        node_embedding_size=args.enc_node_embedding_size,
        edge_embedding_size=args.enc_edge_embedding_size,
        n_layers=args.enc_n_layers,
        gat_n_heads=args.enc_gat_n_heads,
        gat_ffn_ratio=args.enc_gat_ffn_ratio,
        with_fp=False,
        fp_embedding_size=args.enc_fp_embedding_size,
        fp_ffn_size=args.enc_fp_ffn_size,
        fp_gated=args.enc_fp_gated,
        fp_n_heads=args.enc_fp_n_heads,
        fp_size=args.enc_fp_size,
        fp_to_gat_feedback=args.enc_fp_to_gat_feedback,
        gat_to_fp_pooling=args.enc_gat_to_fp_pooling,
    ).to(device)
    decoder = DELRefDecoder(
        node_input_size=args.dec_node_input_size,
        node_emb_size=args.dec_node_emb_size,
        fp_input_size=args.dec_fp_input_size,
        fp_emb_size=args.dec_fp_emb_size,
        output_size=args.dec_output_size,
        output_activation=torch.exp,
        with_fp=False,
        with_dist=False,
    ).to(device)
    model = DELRefNet(encoder, decoder).to(device)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    if args.load_path:
        path = os.path.join(args.load_path, args.name, "model.pt")
        logger.info(f"Loading model from {path}")
        model.load_state_dict(torch.load(path, map_location=device), strict=False)

    class abstract_graph_nn(torch.nn.Module):
        def __init__(self, model):
            super(abstract_graph_nn, self).__init__()
            self.model = model

        def forward(self, x, edge_index, edge_attr, batch):
            return self.model(
            x,
            edge_index,
            edge_attr,
            None,
            batch,
            None,
            None,
        )[0]

    model = abstract_graph_nn(model).to(device)
        

    del_dataset = GraphDataset(
        forced_reload=False,
        target_name=args.target_name,
        fpsize=args.fp_size,
    )

    def get_idx_array(dataset):
        idx_array = []
        pos_idx = []
        for i in tqdm(range(len(dataset))):
            idx_array.append(dataset[i].mol_id.numpy())
            pos_idx.append(i)
        idx_array = np.array(idx_array)
        pos_idx = np.array(pos_idx)
        return idx_array, pos_idx
    
    data_idx = {
        "ca9": np.array([8, 39, 22]),
        "hrp": np.array([4, 96, 74])
    }
    idx_array, pos_idx = get_idx_array(del_dataset)
    data_idx = data_idx[args.target_name]
    def encode(x):
        vec = np.array([120 ** 2, 120, 1])
        return np.sum(x * vec, axis=-1)
    data_idx = pos_idx[np.isin(encode(idx_array), encode(data_idx))][0]
    data = del_dataset[data_idx]
    
    # Explainer
    explainer = Explainer(
        model = model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="node",
            return_type="raw"
        )
    )

    feature = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "atomic_number", "degree", "aromatic", "sp", "sp2", "sp3", "other hybridization", "numhs", "formal_charge", "in_ring"]

    for data in DataLoader([data], batch_size=1, shuffle=False):
        data = data.to(device)
        explanation = explainer(data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        print(f'Generated explanations in {explanation.available_explanations}')

        path = 'feature_importance_hrp.png'
        explanation.visualize_feature_importance(path)
        print(f"Feature importance plot has been saved to '{path}'")

        path = 'subgraph_hrp.pdf'
        explanation.visualize_graph(path)
        print(f"Subgraph visualization plot has been saved to '{path}'")
