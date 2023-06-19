import torch
from torch_geometric.nn import HeteroConv, MessagePassing


class MyMessagePassing(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)

        return out

    def message(self, x_j):
        return x_j


class MyLSTMConv(torch.nn.Module):
    def __init__(self, in_shape, hid_dim, steps):
        super().__init__()
        self.steps = steps
        self.conv = HeteroConv({
            ('cons', 'to', 'vals'): MyMessagePassing(),
            ('vals', 'to', 'cons'): MyMessagePassing(), }, aggr='sum')
        self.lin_vals = torch.nn.Linear(in_shape, hid_dim)
        self.lin_cons = torch.nn.Linear(in_shape, hid_dim)

        self.cell_vals = torch.nn.LSTMCell(hid_dim, hid_dim)
        self.cell_cons = torch.nn.LSTMCell(hid_dim, hid_dim)
        self.norm_vals = torch.nn.BatchNorm1d(hid_dim)
        self.norm_cons = torch.nn.BatchNorm1d(hid_dim)

        self.nn_vals = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(hid_dim, hid_dim),
                                           torch.nn.BatchNorm1d(hid_dim))
        self.nn_cons = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(hid_dim, hid_dim),
                                           torch.nn.BatchNorm1d(hid_dim))

        self.pred_vals = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(hid_dim, 1))

    def forward(self, x_dict, edge_index_dict):
        x_dict['vals'] = torch.relu(self.lin_vals(x_dict['vals']))
        x_dict['cons'] = torch.relu(self.lin_vals(x_dict['cons']))

        h_val = x_dict['vals'].new_zeros(x_dict['vals'].shape)
        c_val = x_dict['vals'].new_zeros(x_dict['vals'].shape)

        h_con = x_dict['cons'].new_zeros(x_dict['cons'].shape)
        c_con = x_dict['cons'].new_zeros(x_dict['cons'].shape)

        hiddens = []
        for i in range(self.steps):
            x_dict = self.conv(x_dict, edge_index_dict)
            x_dict['vals'] = self.nn_vals(x_dict['vals'])
            x_dict['cons'] = self.nn_vals(x_dict['cons'])
            h_val, c_val = self.cell_vals(x_dict['vals'], (h_val, c_val))
            h_con, c_con = self.cell_vals(x_dict['cons'], (h_con, c_con))
            hiddens.append(h_val)

        hiddens = torch.stack(hiddens, dim=0)
        out = self.pred_vals(hiddens)
        return out.squeeze(-1)
