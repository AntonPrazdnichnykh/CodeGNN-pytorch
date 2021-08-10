import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.parameter import Parameter
import math


def calc_degree_matrix_norm(a):
    return torch.diag_embed(torch.pow(a.sum(dim=-1), -0.5))


def create_graph_lapl_norm(a):  # smth from google
    a = a + torch.eye(int(a.size(-1)), device=a.device)
    d_norm = calc_degree_matrix_norm(a)
    return torch.bmm(torch.bmm(d_norm, a), d_norm)   # D^{-1/2} @ (a + 1) @ D^{-1/2}


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: DictConfig):  # in_dim, out_dim, use_bias=True, norm=True):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim, bias=config.use_bias), nn.ReLU())
        self.norm = config.norm
        self.residual = config.residual
        if config.normalization_layer == 'batch_norm':
            self.norm_layer = nn.BatchNorm(out_dim)
        elif config.normalization_layer == 'layer_norm':
            self.norm_layer = nn.LayerNorm(out_dim)
        elif config.normalization_layer == 'none':
            self.norm_layer = nn.Identity()
        else:
            raise NotImplementedError()

    def forward(self, nodes, edges):
        l_norm =  create_graph_lapl_norm(edges) if self.norm else edges + torch.eye(edges.size(-1), device=edges.device)
        out = torch.bmm(l_norm, nodes)  # sums each node vector with its nearest neighbours
        out = self.net(out)
        if self.residual:
            out = out + nodes
        return self.norm_layer(out)


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, is_concat: bool = True, config: DictConfig):
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = config.n_heads

        if is_concat:
            assert out_dim % config.n_heads == 0
            self.hidden_dim = out_dim // config.n_heads
        else:
            self.hidden_dim = out_dim

        self.linear = nn.Linear(in_dim, self.hidden_dim * config.n_heads, bias=False)
        self.attn = nn.Linear(self.hidden_dim * 2, 1, bias=False)

        self.activation = nn.LeakyReLU(negative_slope=config.leaky_relu_negative_slope)
        self.dropout = nn.Dropout(config.dropout)

        self.norm = config.norm
        self.residual = config.residual
        if config.normalization_layer == 'batch_norm':
            self.norm_layer = nn.BatchNorm(out_dim)
        elif config.normalization_layer == 'layer_norm':
            self.norm_layer = nn.LayerNorm(out_dim)
        elif config.normalization_layer == 'none':
            self.norm_layer = nn.Identity()
        else:
            raise NotImplementedError()

    def forward(self, nodes, edges, avg=False):
        batch_size, n_nodes = nodes.shape[:2]
        mapped = self.linear(nodes).view(batch_size, n_nodes, self.n_heads, -1)
        mapped_repeat = mapped.repeat(n_nodes, )  # TODO: copy paste from here: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/459c26b3ed60b810ba0d11fce722d3bf83f699dd/labml_nn/graphs/gat/__init__.py#L36
        edges = edges + torch.eye(edges.size(-1), device=edges.device)
        mapped, attn_weights = self.attn(nodes, edges)
        states = []
        for nodes, weights in zip(mapped, attn_weights):
            state = torch.bmm(weights, nodes)
            if avg:
                states.append(state)
            else:
                states.append(torch.sigmoid(state))
        if avg:
            s = torch.zeros_like(states[0])
            for t in states:
                s = s + t
            final_state = s / len(states)
        else:
            final_state = torch.cat(states, dim=-1)

        return final_state

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.proj_weights = [Parameter(torch.empty(in_dim, out_dim * num_heads))]
        self.attn_weights = Parameter(torch.empty(2 * out_dim * num_heads))
        nn.init.xavier_uniform_(self.proj_weights)
        nn.init.xavier_uniform_(self.attn_weights)

    def forward(self, x, mask):
        mapped = torch.bmm(self.proj_weights, x)
        out_attn_weights, out_mapped = [], []
        for i in range(self.num_heads):
            cur_mapped = mapped[:, :, i * self.out_dim: (i + 1) * self.out_dim]
            mapped_left = torch.cat((cur_mapped, torch.ones_like(cur_mapped)), dim=-1)
            mapped_right = torch.cat((torch.ones_like(cur_mapped), cur_mapped), dim=-1)
            scores = F.leaky_relu(
                torch.bmm(
                    torch.bmm(
                        mapped_left, torch.diag_embed(self.attn_weights[i*2*self.out_dim: (i+1)*2*self.out_dim])
                    ), mapped_right.transpose(1, 2)
                ), negative_slope=0.2
            )
            scores = scores.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(scores, dim=-1)
            out_attn_weights.append(attn_weights)
            out_mapped.append(cur_mapped)
        return out_mapped, out_attn_weights