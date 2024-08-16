from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (LayerNorm, Linear, MultiheadAttention, ReLU, Sequential,
                      Softmax)
from torch_geometric.nn import GATConv, MessagePassing, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

# TODO: free-to-plug model base class with argparse


class DELRefEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        node_embedding_size: int,
        edge_embedding_size: int,
        n_layers: int,
        gat_n_heads: int,
        gat_ffn_ratio: float,
        with_fp: bool,
        fp_size: int,
        fp_embedding_size: int,
        fp_n_heads: int,
        fp_ffn_size: int,
        fp_gated: bool,
        fp_to_gat_feedback: Literal["none", "add", "gate"],
        gat_to_fp_pooling: Literal["mean", "max"],
    ):
        super().__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.fp_embedding_size = fp_embedding_size
        self.n_layers = n_layers
        self.with_fp = with_fp

        self.node_embedding_layer = Linear(node_feat_dim, node_embedding_size)
        self.edge_embedding_layer = Linear(edge_feat_dim, edge_embedding_size)
        self.gat_layers = nn.ModuleList(
            [
                GraphAttnLayer(
                    node_embedding_size, edge_embedding_size, gat_ffn_ratio, gat_n_heads
                )
                for _ in range(n_layers)
            ]
        )
        if with_fp:
            self.fp_size = fp_size
            self.fp_embedding_layer = Linear(fp_size, fp_embedding_size)
            self.fp_enc_layers = nn.ModuleList(
                [
                    AttnLayer(fp_embedding_size, fp_n_heads, fp_ffn_size, fp_gated)
                    for _ in range(n_layers)
                ]
            )
            self.gat_to_fp_layer = nn.ModuleList(
                [self._get_gat_to_fp_layer() for _ in range(n_layers)]
            )
            self.fp_to_gat_feedback = nn.ModuleList(
                [self._get_feedback_layer(fp_to_gat_feedback) for _ in range(n_layers)]
            )
            self.feedback_fn = self._get_feedback_fn(fp_to_gat_feedback)
            self.gat_to_fp_pooling = gat_to_fp_pooling

    def forward(self, node_feat, edge_idx, edge_feat, bb_idx, batch, mol_fp=None):
        # embedding
        node_vec = self.node_embedding_layer(node_feat)
        edge_vec = self.edge_embedding_layer(edge_feat)
        fp_vec = None
        if self.with_fp:
            # mol_fp: (batch_size, bb_num, fp_original_size), make sure fp_size is a divisor of fp_original_size
            mol_fp = mol_fp.view(*mol_fp.shape[:-1], self.fp_size, mol_fp.shape[-1] // self.fp_size).sum(dim=-1)
            # mol_fp: (batch_size, bb_num, fp_size)
            fp_vec = self.fp_embedding_layer(mol_fp)
        # forward
        for i in range(self.n_layers):
            node_vec, edge_vec = self.gat_layers[i](node_vec, edge_idx, edge_vec)
            if self.with_fp:
                fp_vec, _ = self.fp_enc_layers[i](fp_vec)
                fp_vec = fp_vec + self.gat_to_fp_layer[i](
                    self._pooling_with_bb_idx(
                        node_vec, bb_idx, batch, self.gat_to_fp_pooling
                    )
                )
                node_vec = self.feedback_fn(
                    self.fp_to_gat_feedback[i], node_vec, fp_vec, bb_idx, batch
                )
        return node_vec, edge_vec, fp_vec

    def _get_gat_to_fp_layer(self):
        return Sequential(
            Linear(self.node_embedding_size, self.fp_embedding_size), ReLU()
        )

    def _get_feedback_layer(self, layer_type: Literal["none", "add", "gate"]):
        if layer_type == "none":
            return Identity()
        elif layer_type == "add":
            return Sequential(
                Linear(self.fp_embedding_size, self.node_embedding_size),
                ReLU(),
            )
            """
        elif layer_type == "gate":
            return lambda node_vec, fp_vec: node_vec * F.sigmoid(
                Linear(self.fp_embedding_size, self.node_embedding_size)(fp_vec)
            )
            """
        else:
            raise ValueError(f"Invalid feedback layer type: {layer_type}")
        
    def _get_feedback_fn(self, feedback_fn: Literal["none", "add", "gate"]):
        if feedback_fn == "none":
            return self.cal_feedback_none
        elif feedback_fn == "add":
            return self.cal_feedback_add
        else:
            raise ValueError(f"Invalid feedback function: {feedback_fn}")
        
    def cal_feedback_none(self, layer, node_vec, fp_vec, bb_idx, batch):
        return node_vec

    def cal_feedback_add(self, layer, node_vec, fp_vec, bb_idx, batch):
        bb_idx = bb_idx.type(torch.int64)

        bb_idx[bb_idx == 10] = 1    # ! urgent fix for the "feature" in the dataset

        num_bb = bb_idx.max().item()
        fp_emb_size = fp_vec.size()[-1]
        num_node, node_emb_size = node_vec.size()

        idx = batch * num_bb + bb_idx - 1

        fp_vec = fp_vec.view(-1, fp_emb_size)
        fp_2_node_vec = layer(fp_vec)
        """
        node_vec = node_vec + torch.stack(
            [fp_2_node_vec[idx[i]] for i in range(num_node)]
        )
        """
        node_vec = node_vec + fp_2_node_vec[idx]
        node_vec[node_vec.isnan()] = 0.0
        return node_vec

    def _pooling_with_bb_idx(self, hidden, bb_idx, batch, pooling="mean"):
        r"""Seperate the hidden states of building blocks and pool each building block."""
        # hidden: (#sum of nodes in batch, emb_size)
        # bb_idx: (#sum of nodes in batch), range in (1, 2, 3) for current dataset
        # batch: (#sum of nodes in batch)， range in batch_idx
        assert pooling in ("mean", "max")

        bb_idx = bb_idx.type(torch.int64)

        bb_idx[bb_idx == 10] = 1    # ! urgent fix for the "feature" in the dataset

        btz = batch.max().item() + 1
        num_bb = bb_idx.max().item()
        _, emb_size = hidden.size()

        idx = batch * num_bb + bb_idx - 1
        idx_len = btz * num_bb

        """
        if pooling == "mean":
            hidden = torch.stack([hidden[idx == i].mean(dim=0) for i in range(idx_len)])
        elif pooling == "max":
            hidden = torch.stack(
                [hidden[idx == i].max(dim=0).values for i in range(idx_len)]
            )
        """
        if pooling == "mean":
            hidden = gap(hidden, idx, idx_len)
        elif pooling == "max":
            hidden = gmp(hidden, idx, idx_len)
        hidden = hidden.view(btz, num_bb, emb_size)
        hidden[hidden.isnan()] = 0.0
        return hidden


class DELRefDecoder(nn.Module):
    def __init__(
        self,
        node_input_size: int,
        fp_input_size: int,
        node_emb_size: int,
        fp_emb_size: int,
        output_size:int, 
        output_activation=None,
        with_fp: bool = True,
        with_dist: bool = True,
    ):
        super().__init__()
        self.n_in = node_input_size
        self.fp_in = fp_input_size
        self.n_h = node_emb_size
        self.fp_h = fp_emb_size
        self.n_o = output_size

        self.output_activation = (
            get_activation(output_activation)
            if type(output_activation) == str or output_activation is None
            else output_activation
        )
        self.with_fp = with_fp
        self.with_dist = with_dist

        self.n_dec = self._get_node_decoder()
        self.dfl = self._get_dist_feature_layer()
        self.fp_dec = self._get_fp_decoder()
        self.score_head = Linear(self.n_h, self.n_o)

    def forward(self, node_vec, fp_vec, dist_vec, batch):
        # broadcast dist_vec to the same shape as node_vec
        if self.with_dist:
            score = self.n_dec(node_vec) * self.dfl(dist_vec.view(-1, 1))
        else:
            score = self.n_dec(node_vec)
        # average over batch
        score = gap(score, batch)
        score = self.score_head(score)
        score = self.output_activation(score)
        if self.with_fp:
            score = score * self.fp_dec(fp_vec.mean(dim=1))
        return score

    def _get_node_decoder(self):
        return Sequential(
            Linear(self.n_in, self.n_h), ReLU(), Linear(self.n_h, self.n_h), ReLU()
        )

    def _get_dist_feature_layer(self):
        return Sequential(Linear(1, self.n_h), nn.Sigmoid())

    def _get_fp_decoder(self):
        return Sequential(
            Linear(self.fp_in, self.fp_h), ReLU(), Linear(self.fp_h, self.n_o), nn.Sigmoid()
        )


class DELRefNet(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, node_feat, edge_idx, edge_feat, bb_idx, batch, mol_fp, dist_feat):
        node_vec, edge_vec, fp_vec = self.encoder(
            node_feat, edge_idx, edge_feat, bb_idx, batch, mol_fp
        )
        score = self.decoder(node_vec, fp_vec, dist_feat, batch)
        return score


class GraphAttnLayer(MessagePassing):
    def __init__(
        self,
        node_embedding_size: int,
        edge_embedding_size: int,
        ffn_ratio: float = 4.0,
        n_heads: int = 1,
    ):
        super().__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.ffn_ratio = ffn_ratio
        self.node_ffn_size = int(self.node_embedding_size * self.ffn_ratio)
        self.edge_ffn_size = int(self.edge_embedding_size * self.ffn_ratio)
        self.n_heads = n_heads

        assert self.node_embedding_size % self.n_heads == 0

        self.gat_conv = GATConv(
            node_embedding_size,
            node_embedding_size // n_heads,
            n_heads,
            edge_dim=edge_embedding_size,
        )

        self.n2e_lin = Linear(node_embedding_size, edge_embedding_size)
        self.node_post_attn_layers = PostAttnLayer(
            self.node_embedding_size, self.node_ffn_size
        )
        self.edge_post_attn_layers = PostAttnLayer(
            self.edge_embedding_size, self.edge_ffn_size
        )

    def forward(self, node_vec, edge_idx, edge_vec):
        node_update = self.gat_conv(x=node_vec, edge_index=edge_idx, edge_attr=edge_vec)
        # edge update
        # first, calculate the edge update from node
        # second, add the edge update to the edge_vec according to the edge_idx
        edge_update = self.n2e_lin(node_update)
        edge_update = edge_update[edge_idx].sum(dim=0)
        edge_vec = edge_vec + edge_update
        # post attn
        node_vec = self.node_post_attn_layers(node_vec, node_update)
        edge_vec = self.edge_post_attn_layers(edge_vec, edge_update)
        return node_vec, edge_vec


class PostAttnLayer(nn.Module):
    def __init__(self, embedding_size: int, ffn_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.ffn = FFN(self.embedding_size, self.ffn_size)
        self.layer_norm = nn.ModuleList(
            [LayerNorm(self.embedding_size) for _ in range(2)]
        )

    def forward(self, x, attn):
        x = self.layer_norm[0](x + attn)
        x = self.layer_norm[1](x + self.ffn(x))
        return x


class FPAttnEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channel: int,
        embedding_size: int,
        ffn_size: int,
        n_attn_heads: int,
        n_layers: int,
        gated: bool = True,
    ):
        super().__init__()

        self.emb_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers

        self.embedding_layer = Linear(
            in_features=in_channel, out_features=embedding_size
        )
        self.attn_layers = nn.ModuleList(
            [
                AttnLayer(
                    embedding_size=embedding_size,
                    n_attn_heads=n_attn_heads,
                    ffn_size=ffn_size,
                    gated=gated,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, mol_fp: torch.Tensor):
        # (batch_size, bb_num, fp_size)
        x = self.embedding_layer(mol_fp)
        # (batch_size, bb_num, emb_size)
        for layer_idx in range(x):
            x, _ = self.attn_layers[layer_idx](x)
        return x


class AttnLayer(torch.nn.Module):
    def __init__(
        self,
        embedding_size: int,
        n_attn_heads: int,
        ffn_size: int,
        gated: bool = True,
        **kwargs,
    ):
        # TODO positional encoding
        super().__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.gated = gated

        self.multi_head_attn = MultiheadAttention(
            embed_dim=embedding_size, num_heads=n_attn_heads, **kwargs
        )
        self.kqv_linear_layer = Linear(embedding_size, embedding_size * 3, bias=False)

        if self.gated:
            self.gate_linear_layer = Linear(embedding_size, embedding_size)

        self.ffn = Sequential(
            Linear(self.embedding_size, self.ffn_size),
            ReLU(),
            Linear(self.ffn_size, self.embedding_size),
        )
        self.post_attn_layers = PostAttnLayer(self.embedding_size, self.ffn_size)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        Q, K, V = torch.split(self.kqv_linear_layer(x), self.embedding_size, dim=-1)
        attn, attn_weight = self.multi_head_attn(Q, K, V)
        if self.gated:
            attn = attn * F.sigmoid(self.gate_linear_layer(x))
        else:
            pass
        x = torch.transpose(x, 0, 1)
        attn = torch.transpose(attn, 0, 1)
        x = self.post_attn_layers(x, attn)
        return x, attn_weight


class FFN(torch.nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layer = Sequential(
            Linear(self.embedding_size, self.hidden_size),
            ReLU(),
            Linear(self.hidden_size, self.embedding_size),
        )

    def forward(self, x):
        return self.layer(x)


class RegressionHead(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        output_activation=None,
    ):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.output_activation = get_activation(output_activation)

    def forward(self, x):
        x = self.lin1(x)
        x = ReLU()(x)
        x = self.lin2(x)
        x = self.output_activation(x)
        return x


class ClassificationHead(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.relu = ReLU()

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = Softmax(dim=-1)(x)
        return x
    
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


def get_activation(activation: Optional[str] = "relu"):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == None:
        # identity
        return lambda x: x
    else:
        raise ValueError(f"Invalid activation: {activation}")
