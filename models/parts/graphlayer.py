import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig



def calc_degree_matrix_norm(a):
    return torch.diag_embed(torch.pow(a.sum(dim=-1), -0.5))


def create_graph_lapl_norm(a):
    a = a + torch.eye(int(a.size(-1)), device=a.device)
    d_norm = calc_degree_matrix_norm(a)
    return torch.bmm(torch.bmm(d_norm, a), d_norm)   # D^{-1/2} @ (a + 1) @ D^{-1/2}


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: DictConfig, *args, **kwargs):  # in_dim, out_dim, use_bias=True, norm=True):
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
    """
    Reference code: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/459c26b3ed60b810ba0d11fce722d3bf83f699dd/labml_nn/graphs/gat/__init__.py#L36
    """
    def __init__(self, in_dim: int, out_dim: int, config: DictConfig, is_concat: bool = True):
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = config.n_heads

        self.hidden_dim = out_dim # // config.n_heads
        # if is_concat:
        #     assert out_dim % config.n_heads == 0
        #     self.hidden_dim = out_dim // config.n_heads
        # else:
        #     self.hidden_dim = out_dim

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

    def forward(self, nodes, edges):
        batch_size, n_nodes = nodes.shape[:2]
        mapped = self.linear(nodes).view(batch_size, n_nodes, self.n_heads, -1)
        mapped_repeat = mapped.repeat(1, n_nodes, 1, 1)
        mapped_repeat_interleave = mapped.repeat_interleave(n_nodes, dim=1)
        mapped_concat = torch.cat((mapped_repeat_interleave, mapped_repeat), dim=-1)
        mapped_concat = mapped_concat.view(batch_size, n_nodes, n_nodes, self.n_heads, 2 * self.hidden_dim)

        scores = self.activation(self.attn(mapped_concat)).squeeze(-1)
        # edges = edges.unsqueeze(-1)

        scores = scores.masked_fill(edges == 0, float('-inf'))
        attn_weights = self.dropout(F.softmax(scores, dim=2))
        attn_res = torch.einsum('kijh, kjhf->kihf', attn_weights, mapped)

        if self.is_concat:
            return torch.sigmoid(attn_res.reshape(batch_size, self.n_heads * self.hidden_dim))
        return torch.sigmoid(attn_res.mean(dim=2))
