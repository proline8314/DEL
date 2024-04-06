from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn import Linear, MultiheadAttention, ReLU, Sequential
from torch_geometric.nn import GATConv, MessagePassing, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

# TODO: free-to-plug model base class with argparse

class DELRefEncoder(torch.nn.Module): 
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        node_embedding_size: int,
        edge_embedding_size: int,
        gat_n_heads: int,
        n_layers: int,
        with_fp: bool,
        fp_size: int,
        fp_embedding_size: int,
        fp_n_heads: int,
        fp_n_layers: int,
        fp_gated: bool,
        fp_to_gat_feedback: Literal["none", "concat", "add", "gate", ""],
    ):
        super().__init__()
        self.node_embedding_layer = Linear(node_feat_dim, node_embedding_size)
        self.edge_embedding_layer = Linear(edge_feat_dim, edge_embedding_size)
        self.gat_layers = [GraphAttnLayer(node_embedding_size, edge_embedding_size, gat_n_heads) for _ in range(n_layers)]
        if with_fp:
            self.fp_embedding_layer = Linear(fp_size, fp_embedding_size)
            self.fp_enc_layers = [AttnLayer(fp_embedding_size, fp_n_heads, fp_gated) for _ in range(fp_n_layers)]



class GraphAttnLayer(MessagePassing):
    def __init__(
        self, node_embedding_size: int, edge_embedding_size: int, n_heads: int = 1
    ):
        super().__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.n_heads = n_heads
        assert self.node_embedding_size % self.n_heads == 0

        self.gat_conv = GATConv(
            node_embedding_size,
            node_embedding_size // n_heads,
            n_heads,
            edge_dim=edge_embedding_size,
        )
        self.n2e_lin = Linear(node_embedding_size, edge_embedding_size)
        self.relu = ReLU()

    def forward(self, node_vec, edge_idx, edge_vec):
        node_vec = self.gat_conv(node_vec, edge_idx, edge_attr=edge_vec)
        # edge update
        # first, calculate the edge update from node
        # second, add the edge update to the edge_vec according to the edge_idx
        edge_update = self.n2e_lin(node_vec)
        edge_update = edge_update[edge_idx].sum(dim=1)
        edge_vec = edge_vec + edge_update

        node_vec = self.relu(node_vec)
        edge_vec = self.relu(edge_vec)
        return node_vec, edge_vec


class FPAttnEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channel: int,
        embedding_size: int,
        n_attn_heads: int,
        n_layers: int,
        gated: bool = True,
    ):
        super().__init__()

        self.emb_size = embedding_size
        self.n_layers = n_layers

        self.embedding_layer = Linear(
            in_features=in_channel, out_features=embedding_size
        )
        self.attn_layers = [
            AttnLayer(
                embedding_size=embedding_size, n_attn_heads=n_attn_heads, gated=gated
            )
            for _ in range(n_layers)
        ]

    def forward(self, mol_fp: torch.Tensor):
        # (batch_size, bb_num, fp_size)
        x = self.embedding_layer(mol_fp)
        # (batch_size, bb_num, emb_size)
        for layer_idx in range(x):
            x, _ = self.attn_layers[layer_idx](x)
        return x


class AttnLayer(torch.nn.Module):
    def __init__(
        self, embedding_size: int, n_attn_heads: int, gated: bool = True, **kwargs
    ):
        # TODO positional encoding
        super().__init__()

        self.embedding_size = embedding_size
        self.gated = gated

        self.multi_head_attn = MultiheadAttention(
            embed_dim=embedding_size, num_heads=n_attn_heads, **kwargs
        )
        self.kqv_linear_layer = Linear(embedding_size, embedding_size * 3, bias=False)

        if self.gated:
            self.gate_linear_layer = Linear(embedding_size, embedding_size)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        Q, K, V = torch.split(self.kqv_linear_layer(x), self.embedding_size, dim=-1)
        attn, attn_weight = self.multi_head_attn(Q, K, V)
        if self.gated:
            x = attn * F.sigmoid(self.gate_linear_layer(x))
        else:
            x = attn
        x = torch.transpose(x, 0, 1)
        return x, attn_weight


class RegressionHead(torch.nn.Module):
    def __init__(self):
        pass


class ClassificationHead(torch.nn.Module):
    def __init__(self):
        pass
