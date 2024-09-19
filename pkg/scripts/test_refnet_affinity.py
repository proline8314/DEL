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
from torch_geometric.loader import DataLoader
from tqdm import tqdm

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up arguments
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--name", type=str, default="ca9_large")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--load_path", type=str, default="/data02/gtguo/DEL/data/weights/refnet/")
    # dataset
    parser.add_argument("--target_name", type=str, default="ca9")
    parser.add_argument("--fp_size", type=int, default=2048)
    parser.add_argument("--forced_reload", action="store_true")

    # data split
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--valid_size", type=float, default=0.2)
    # dataloader
    parser.add_argument("--num_workers", type=int, default=8)
    # model encoder
    parser.add_argument("--enc_node_feat_dim", type=int, default=19)
    parser.add_argument("--enc_edge_feat_dim", type=int, default=2)
    parser.add_argument("--enc_node_embedding_size", type=int, default=64)
    parser.add_argument("--enc_edge_embedding_size", type=int, default=64)
    parser.add_argument("--enc_n_layers", type=int, default=3)
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

    # record
    parser.add_argument(
        "--record_path", type=str, default="/data02/gtguo/DEL/data/records/refnet/"
    )

    args = parser.parse_args()

    # print arguments to log
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # Set up tensorboard
    tb_path = os.path.join(
        os.path.dirname(__file__), os.path.join(args.record_path, args.name)
    )
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    writer = SummaryWriter(tb_path)

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
        with_fp=args.enc_with_fp,
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
        with_fp=args.dec_with_fp,
        with_dist=args.dec_with_dist,
    ).to(device)
    model = DELRefNet(encoder, decoder).to(device)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Datasets
    del_dataset = GraphDataset(
        forced_reload=args.forced_reload,
        target_name=args.target_name,
        fpsize=args.fp_size,
    )
    
    def get_idx_array(dataset):
        idx_array = []
        for i in tqdm(range(len(dataset))):
            idx_array.append(dataset[i].mol_id.numpy())
        idx_array = np.array(idx_array)
        return idx_array
    
    idx_array = get_idx_array(del_dataset)
        

    del_dataloader = DataLoader(
        del_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Load model
    if args.load_path:
        path = os.path.join(args.load_path, args.name, "model.pt")
        logger.info(f"Loading model from {path}")
        model.load_state_dict(torch.load(path, map_location=device))

    # Test model
    # output target lambda

    model.eval()
    target_lambda = []
    target_lambda_wo_dist = []
    target_lambda_wo_fp = []

    for i, data in tqdm(enumerate(del_dataloader)):
        data = data.to(device)
        output = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.node_bbidx,
            data.batch,
            data.bbfp.view(data.batch_size, -1, args.fp_size),
            data.node_dist,
        )
        output = output.detach().cpu()
        target_lambda.append(output[:, 0])

    target_lambda = torch.cat(target_lambda, dim=0).numpy()

    # split data

    def split_data(data, idx_array, target_name):
        print(len(data), len(idx_array))

        if target_name == "ca9":
            array_first = []
            array_second = []
            array_third = []
            for i in range(idx_array.shape[0]):
                if idx_array[i, -1] == 22:
                    array_first.append(data[i])
                elif idx_array[i, -1] == 39:
                    array_second.append(data[i])
                else:
                    array_third.append(data[i])

            array_first = np.array(array_first)
            array_second = np.array(array_second)
            array_third = np.array(array_third)

            return array_first, array_second, array_third
        
        elif target_name == "hrp":
            array_first = []
            array_second = []
            array_third = []
            array_forth = []
            for i in range(idx_array.shape[0]):
                if idx_array[i, 1] == 96:
                    array_first.append(data[i])
                elif idx_array[i, 1] == 30:
                    array_second.append(data[i])
                elif idx_array[i, 1] == 18:
                    array_third.append(data[i])
                else:
                    array_forth.append(data[i])

            array_first = np.array(array_first)
            array_second = np.array(array_second)
            array_third = np.array(array_third)
            array_forth = np.array(array_forth)

            return array_first, array_second, array_third, array_forth
        else: 
            raise ValueError("Invalid target name")
        
    def split_array_with_scaffold(array, idx_array):
        output_dict = {}
        for i in range(idx_array.shape[0]):
            if idx_array[i, 0] not in output_dict:
                output_dict[idx_array[i, 0]] = [array[i]]
            else:
                output_dict[idx_array[i, 0]].append(array[i])
        for key in output_dict:
            output_dict[key] = np.array(output_dict[key])
        return output_dict
    
    logger.info("Splitting data")

    output_dict = split_array_with_scaffold(target_lambda, idx_array)

    splited_dict = {}
    for key in output_dict:
        splited_dict[key] = split_data(output_dict[key], idx_array[idx_array[:, 0] == key], args.target_name)

    # save data

    def save_data(data, path):
        with open(path, "wb") as f:
            pickle.dump(data, f)

    data_dir = "/data02/gtguo/DEL/data/temp"
    data_path = os.path.join(data_dir, args.name)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    logger.info("Saving data")

    save_data(splited_dict, os.path.join(data_path, "target_lambda.pkl"))
