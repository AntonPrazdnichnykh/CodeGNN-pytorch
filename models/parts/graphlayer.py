import torch
import torch.nn as nn


def calc_degree_matrix_norm(a):
    return torch.diag_embed(torch.pow(a.sum(dim=-1), -0.5))


def create_graph_lapl_norm(a):
    a = a + torch.eye(int(a.size(-1)))
    d_norm = calc_degree_matrix_norm(a)
    return torch.bmm(torch.bmm(d_norm, a), d_norm)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_bias=True):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim, bias=use_bias), nn.ReLU())

    def forward(self, nodes, edges):
        # assert isinstance(x, list)
        l_norm = create_graph_lapl_norm(edges)  #  edges + torch.eye(edges.size(1))  #
        out = torch.bmm(l_norm, nodes)
        return self.net(out)

    # def compute_out_shape(self, input_shape):
    #     assert isinstance(input_shape, list)
    #     return None, input_shape[0][1], self.out_dim
