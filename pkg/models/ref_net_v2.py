from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (LayerNorm, Linear, MultiheadAttention, ReLU, Sequential,
                      Softmax)
from torch_geometric.data import Data
from torch_geometric.nn import (GATConv, MessagePassing, MulAggregation,
                                SumAggregation, TopKPooling)
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


class RefNetV2(nn.Module):
    def __init__(
        self,
        synthon_feat_input: "PyGDataInputLayer",
        molecule_feat_input: "PyGDataInputLayer",
        synthon_encoder: "GraphAttnEncoder",
        molecule_encoder: "GraphAttnEncoder",
        bidirectional_pipes: List["BidirectionalPipe"],
        reaction_yield_head: "RegressionHead",
        affinity_head: "RegressionHead",
    ):
        super().__init__()

        self.synthon_feat_input = synthon_feat_input
        self.molecule_feat_input = molecule_feat_input
        self.synthon_encoder = synthon_encoder
        self.molecule_encoder = molecule_encoder
        self.bidirectional_pipe = bidirectional_pipes
        self.reaction_yield_head = reaction_yield_head
        self.affinity_head = affinity_head

        self.mul_agg = MulAggregation()
        self.sum_agg = SumAggregation()
        self.softplus = nn.Softplus()

        # * make sure the synthon encoder, molecule encoder, and bidirectional pipe have the same number of layers
        assert (
            synthon_encoder.num_layers
            == molecule_encoder.num_layers
            == len(bidirectional_pipes)
        ), "The number of layers in synthon encoder, molecule encoder, and bidirectional pipe should be the same."

        self.n_layers = synthon_encoder.num_layers

    def forward(self, synthon_data: Data, molecule_data: Data):

        # * get the node and edge features from the input data
        synthon_node_vec, synthon_edge_idx, synthon_edge_vec = self.synthon_feat_input(
            synthon_data
        )
        molecule_node_vec, molecule_edge_idx, molecule_edge_vec = (
            self.molecule_feat_input(molecule_data)
        )

        # * encode the synthon and molecule graphs
        for i in range(self.n_layers):
            synthon_node_vec, synthon_edge_vec = self.synthon_encoder.layers[i](
                synthon_node_vec, synthon_edge_idx, synthon_edge_vec
            )
            molecule_node_vec, molecule_edge_vec = self.molecule_encoder.layers[i](
                molecule_node_vec, molecule_edge_idx, molecule_edge_vec
            )
            synthon_node_update = self.bidirectional_pipe[i].child2parent(
                synthon_node_vec,
                molecule_node_vec,
                synthon_data.batch,
                molecule_data.batch,
                molecule_data.synthon_index,
                molecule_data.ptr
            )
            molecule_node_update = self.bidirectional_pipe[i].parent2child(
                synthon_node_vec,
                molecule_node_vec,
                synthon_data.batch,
                molecule_data.batch,
                molecule_data.synthon_index,
                molecule_data.ptr
            )
            synthon_node_vec = synthon_node_vec + synthon_node_update
            molecule_node_vec = molecule_node_vec + molecule_node_update

            synthon_node_vec[synthon_node_vec.isnan()] = 0.0
            molecule_node_vec[molecule_node_vec.isnan()] = 0.0

        # * get the reaction yield and affinity predictions
        yield_pred = self.get_reaction_yield(
            synthon_edge_vec, synthon_edge_idx, synthon_data.batch
        )
        # affinity_pred: (batch_size, 2)
        affinity_pred = self.get_affinity(molecule_node_vec, molecule_data.batch)

        affinity_pred = yield_pred * affinity_pred
        return affinity_pred

    def get_reaction_yield(self, synthon_edge_vec, synthon_edge_idx, synthon_batch):
        # synthon_edge_vec: (num_edges * 2, edge_embedding_size)
        # * average over the neighboring edge vectors
        btz, _ = synthon_edge_vec.shape
        synthon_edge_batch = self._get_edge_batch(synthon_edge_idx, synthon_batch)

        synthon_edge_vec = synthon_edge_vec.view(btz // 2, 2, -1)[:, 0, :]
        synthon_edge_batch = synthon_edge_batch[::2]

        # * calculate the reaction yield
        yield_pred = self.reaction_yield_head(synthon_edge_vec)

        # * aggregate the reaction yield predictions
        # yield_pred: (batch_size, 1)
        yield_pred = self.mul_agg(yield_pred, synthon_edge_batch)
        return yield_pred

    def get_affinity(self, molecule_node_vec, molecule_batch):
        # molecule_node_vec: (num_nodes, node_embedding_size)
        # * calculate pairwise affinity
        # affinity_pred: (num_nodes, 2)
        affinity_pred = self.affinity_head(molecule_node_vec)

        # * aggregate the affinity (free energy) predictions
        # affinity_pred: (batch_size, 2)
        affinity_pred = self.sum_agg(affinity_pred, molecule_batch)

        # * convert the energy to equilibrium constant
        affinity_pred = self.softplus(affinity_pred)
        return affinity_pred

    def _get_edge_batch(self, edge_idx, batch):
        return batch[edge_idx[0]]


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


class GraphAttnEncoder(nn.Module):
    def __init__(
        self,
        node_embedding_size: int,
        edge_embedding_size: int,
        ffn_ratio: float = 4.0,
        n_heads: int = 4,
        num_layers: int = 5,
    ):
        super().__init__()
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.ffn_ratio = ffn_ratio
        self.n_heads = n_heads
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                GraphAttnLayer(
                    node_embedding_size,
                    edge_embedding_size,
                    ffn_ratio=ffn_ratio,
                    n_heads=n_heads,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, node_vec, edge_idx, edge_vec):
        for layer in self.layers:
            node_vec, edge_vec = layer(node_vec, edge_idx, edge_vec)
        return node_vec, edge_vec


class BidirectionalPipe(nn.Module):
    """
    Aggregrate the molecule graph information into the synthon graph,
    and distribute the synthon graph information into the molecule graph.
    """

    def __init__(
        self,
        parent_embedding_size: int,
        child_embedding_size: int,
        hidden_ratio: float = 4.0,
        num_families: int = 4,
    ):
        super().__init__()
        self.parent_embedding_size = parent_embedding_size
        self.child_embedding_size = child_embedding_size
        self.hidden_ratio = hidden_ratio
        self.num_family = num_families

        self.parent_to_child_layer = Sequential(
            Linear(parent_embedding_size, int(parent_embedding_size * hidden_ratio)),
            ReLU(),
            Linear(int(parent_embedding_size * hidden_ratio), child_embedding_size),
        )

        self.child_to_parent_layer = Sequential(
            Linear(child_embedding_size, int(child_embedding_size * hidden_ratio)),
            ReLU(),
            Linear(int(child_embedding_size * hidden_ratio), parent_embedding_size),
        )

    def parent2child(self, parent, child, parent_batch, child_batch, connection, ptr):
        # ? connection = connection.type(torch.int64)
        connection = (connection - ptr[child_batch]).type(torch.int64)

        # batched_connection: (num_child_nodes,), the index of batched parent nodes
        batched_connection = child_batch * self.num_family + connection
        connection_mask = connection == -1

        # parent: (num_parent_nodes, parent_embedding_size)
        # parent_batch: (num_parent_nodes,)
        child_update = self.parent_to_child_layer(parent)

        # child_update: (num_child_nodes, child_embedding_size)
        child_update = child_update[batched_connection]
        child_update[connection_mask] = 0.0
        child_update[child_update.isnan()] = 0.0
        return child_update

    def child2parent(self, parent, child, parent_batch, child_batch, connection, ptr):
        # ? connection = connection.type(torch.int64)
        connection = (connection - ptr[child_batch]).type(torch.int64)

        # batched_connection: (num_child_nodes,), the index of batched parent nodes
        batched_connection = child_batch * self.num_family + connection
        connection_mask = connection == -1

        # parent: (num_parent_nodes, parent_embedding_size)
        parent_update = self.child_to_parent_layer(child)

        # parent_update: (num_child_nodes w/ mask, parent_embedding_size)
        parent_update = parent_update[connection_mask == False]
        batched_connection = batched_connection[connection_mask == False]

        batch_size = parent_batch.max().item() + 1
        num_batched_idx = batch_size * self.num_family

        parent_update = gap(parent_update, batched_connection, num_batched_idx)
        parent_update[parent_update.isnan()] = 0.0
        return parent_update


class PyGDataInputLayer(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        node_embedding_size: int,
        node_embedding_type: Literal["Linear", "Embedding", "Embedding-Linear"],
        edge_feat_dim: Optional[int],
        edge_embedding_size: int,
        edge_embedding_type: Literal["Linear", "Embedding", "Embedding-Linear", "None"],
        *,
        token_size: int = 8,
    ):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.node_embedding_size = node_embedding_size
        self.node_embedding_type = node_embedding_type

        self.edge_feat_dim = edge_feat_dim
        self.edge_embedding_size = edge_embedding_size
        self.edge_embedding_type = edge_embedding_type

        self.token_size = token_size

        self.node_embedding_layer = self._get_node_embedding_layer()
        self.edge_embedding_layer = self._get_edge_embedding_layer()

    def forward(self, data: Data):
        node_feat = data.x
        edge_idx = data.edge_index
        node_vec = self.node_embedding_layer(node_feat)

        if self.edge_embedding_type == "None":
            edge_vec = self._get_default_rep_vector(
                (edge_idx.shape[-1], self.edge_embedding_size), node_feat.device
            )
        else:
            edge_feat = data.edge_attr
            edge_vec = self.edge_embedding_layer(edge_feat)

        return node_vec, edge_idx, edge_vec

    def _get_node_embedding_layer(self) -> nn.Module:
        if self.node_embedding_type == "Linear":
            return self._get_linear_layer(self.node_feat_dim, self.node_embedding_size)
        elif self.node_embedding_type == "Embedding":
            return self._get_bitvec_embedding_layer(
                self.node_feat_dim, self.token_size, self.node_embedding_size
            )
        elif self.node_embedding_type == "Embedding-Linear":
            return nn.Sequential(
                self._get_bitvec_embedding_layer(
                    self.node_feat_dim, self.token_size, self.node_feat_dim // self.token_size
                ),
                self._get_linear_layer(self.node_feat_dim // self.token_size, self.node_embedding_size),
            )
        else:
            raise ValueError(f"Invalid node embedding type: {self.node_embedding_type}")

    def _get_edge_embedding_layer(self) -> nn.Module:
        if self.edge_embedding_type == "Linear":
            return self._get_linear_layer(self.edge_feat_dim, self.edge_embedding_size)
        elif self.edge_embedding_type == "Embedding":
            return self._get_bitvec_embedding_layer(
                self.edge_feat_dim, self.token_size, self.edge_embedding_size
            )
        elif self.edge_embedding_type == "Embedding-Linear":
            return nn.Sequential(
                self._get_bitvec_embedding_layer(
                    self.edge_feat_dim, self.token_size, self.edge_feat_dim // self.token_size
                ),
                self._get_linear_layer(self.edge_feat_dim // self.token_size, self.edge_embedding_size),
            )
        elif self.edge_embedding_type == "None":
            return Identity()
        else:
            raise ValueError(f"Invalid edge embedding type: {self.edge_embedding_type}")

    def _get_linear_layer(self, in_size, out_size) -> nn.Module:
        return Linear(in_size, out_size)

    def _get_bitvec_embedding_layer(
        self, bitvec_dim, token_size, embedding_size
    ) -> nn.Module:
        return BitVecEmbedding(bitvec_dim, token_size, embedding_size)

    def _get_default_rep_vector(self, size, device) -> torch.Tensor:
        return torch.zeros(size, device=device)


class BitVecEmbedding(nn.Module):
    def __init__(self, bitvec_dim: int, token_size: int, embedding_size: int):
        super().__init__()

        self.bitvec_dim = bitvec_dim
        self.token_size = token_size

        assert bitvec_dim % token_size == 0

        self.num_tokens = bitvec_dim // token_size

        assert embedding_size % self.num_tokens == 0

        self.embedding_size = embedding_size
        self.token_embedding_size = embedding_size // self.num_tokens

        self.embedding = nn.Embedding(2**token_size, self.token_embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., bitvec_dim)
        x = x.view(*x.shape[:-1], self.num_tokens, self.token_size)

        # x: (..., num_tokens, token_size)
        x = self.encode_bits(x)

        # x: (..., num_tokens)
        x = self.embedding(x)

        # x: (..., num_tokens, token_embedding_size)
        x = x.view(*x.shape[:-2], -1)

        # x: (..., embedding_size)
        return x

    def encode_bits(self, x: torch.Tensor) -> torch.LongTensor:
        """
        Convert a tensor of bits to a tensor of integers.
        """
        x = x.long()

        # x: (..., token_size)
        x = x * (2 ** torch.arange(self.token_size, device=x.device, dtype=x.dtype)).repeat(*x.shape[:-1], 1)
        x = x.sum(dim=-1)

        # x: (...,)
        return x


class RegressionHead(nn.Module):
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


class ClassificationHead(nn.Module):
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
        return Identity()
    else:
        raise ValueError(f"Invalid activation: {activation}")
