import torch
import torch.nn.functional as F

from models.utils import scatter_sum, MLP
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class GCNConv(MessagePassing):
    def __init__(self, in_dim, edge_dim, hid_dim, num_mlp_layers, norm, in_place = True):
        super(GCNConv, self).__init__(aggr='add')

        self.in_place = in_place
        self.lin_src = torch.nn.Linear(in_dim, hid_dim)
        self.lin_dst = torch.nn.Linear(in_dim, hid_dim)
        self.lin_edge = torch.nn.Linear(edge_dim, hid_dim)
        self.mlp = MLP([hid_dim] * (num_mlp_layers + 1), norm=norm)

    def forward(self, x, edge_index, edge_attr):
        x = (self.lin_src(x[0]), x[1])

        if edge_attr is not None and hasattr(self, 'lin_edge'):
            edge_attr = self.lin_edge(edge_attr)

        row, col = edge_index
        deg_src = degree(row, x[0].shape[0], dtype = x[0].dtype) + 1
        deg_src_inv_sqrt = deg_src.pow(-0.5)
        deg_src_inv_sqrt[deg_src_inv_sqrt == float('inf')] = 0

        deg_dst = degree(col, x[1].shape[0], dtype = x[1].dtype) + 1
        deg_dst_inv_sqrt = deg_dst.pow(-0.5)
        deg_dst_inv_sqrt[deg_dst_inv_sqrt == float('inf')] = 0

        norm = deg_src_inv_sqrt[row] * deg_dst_inv_sqrt[col]

        if not self.in_place:
            msg = norm.view(-1, 1) * F.relu(x[0][row] + edge_attr)
            out = scatter_sum(msg, col, dim=0)
        else:
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

        x_dst = x[1]
        x_dst = self.lin_dst(x_dst)
        out = out + x_dst

        return self.mlp(out)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
