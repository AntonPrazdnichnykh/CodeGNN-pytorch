import torch
from torch import nn
from .graphlayer import GCNLayer
from omegaconf import DictConfig


class GCNEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        if config.gcn_encoder.rnn_cell == 'gru':
            self.rnn_cell = nn.GRUCell(config.gcn_hidden_size, config.hidden_size)
        elif config.gcn_encoder.rnn_cell == 'lstm':
            self.rnn_cell = nn.LSTMCell(config.gcn_hidden_size, config.hidden_size)
        elif config.hcn_encoder.rnn_cell == 'none':
            self.rnn_cell = None
        else:
            raise NotImplementedError()

        if self.rnn_cell:
            self.hidden_size = config.hidden_size
            self.hidden_init = nn.Linear(config.embedding_size, config.gcn_hidden_size)
            self.layers = nn.ModuleList(
                [
                    GCNLayer(
                        config.gcn_hidden_size, config.gcn_hidden_size, config.gcn_encoder
                    ) for _ in range(config.num_hops)
                ]
            )
        else:
            layers = [GCNLayer(config.embedding_size, config.gcn_hidden_size, config.gcn_encoder)]
            layers.extend(
                [GCNLayer(config.gcn_hidden_size, config.gcn_hidden_size, config.gcn_encoder) for _ in range(config.num_hops - 1)]
            )
            self.layers = nn.ModuleList(layers)

    def forward(self, embedded_nodes, edges):
        if self.rnn_cell:
            ast_enc = self.rnn_cell(
                self.hidden_init(embedded_nodes), embedded_nodes.new_zeros((embedded_nodes.size(0), self.hidden_size))
            )
            h_prev = ast_enc.clone()
        else:
            ast_enc = embedded_nodes

        for layer in self.layers:
            ast_enc = layer(ast_enc, edges)
            if self.rnn_cell:
                ast_enc = self.rnn_cell(ast_enc, h_prev)
                h_prev = ast_enc.clone()
        return ast_enc


class GATEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        pass

    def forward(self, embedded_nodes, edges):
        pass
