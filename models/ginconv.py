import torch

from models.utils import scatter_sum, MLP
from torch_geometric.nn import MessagePassing


class GINEConv(MessagePassing):
    def __init__(self, in_dim, edge_dim, hid_dim, num_mlp_layers, norm, in_place = True):
        super(GINEConv, self).__init__(aggr="add")

        self.in_place = in_place
        self.lin_src = torch.nn.Linear(in_dim, hid_dim)
        self.lin_dst = torch.nn.Linear(in_dim, hid_dim)
        self.lin_edge = torch.nn.Linear(edge_dim, hid_dim)
        self.mlp = MLP([hid_dim] * (num_mlp_layers + 1), norm=norm)
        self.eps = torch.nn.Parameter(torch.Tensor([1.]))

    def forward(self, x, edge_index, edge_attr):
        x = (self.lin_src(x[0]), x[1])

        if edge_attr is not None and hasattr(self, 'lin_edge'):
            edge_attr = self.lin_edge(edge_attr)

        row, col = edge_index

        if not self.in_place:
            msg = torch.relu(x[0][row] + edge_attr)
            out = scatter_sum(msg, col, dim=0)
        else:
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        x_dst = (1 + self.eps) * x[1]
        x_dst = self.lin_dst(x_dst)
        out = out + x_dst

        return self.mlp(out)

    def message(self, x_j, edge_attr):
        m = torch.relu(x_j + edge_attr)
        return m

    def update(self, aggr_out):
        return aggr_out
