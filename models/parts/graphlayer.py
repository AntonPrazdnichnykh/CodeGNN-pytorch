import torch
import torch.nn as nn
from omegaconf import DictConfig


def calc_degree_matrix_norm(a):
    return torch.diag_embed(torch.pow(a.sum(dim=-1), -0.5))


def create_graph_lapl_norm(a):  # smth from google
    a = a + torch.eye(int(a.size(-1)), device=a.device)
    d_norm = calc_degree_matrix_norm(a)
    return torch.bmm(torch.bmm(d_norm, a), d_norm)   # D^{-1/2} @ (a + 1) @ D^{-1/2}


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: DictConfig):  # in_dim, out_dim, use_bias=True, norm=True):
        super().__init__()
        self.out_dim = out_dim
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

