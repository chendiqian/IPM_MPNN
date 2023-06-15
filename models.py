from torch_geometric.nn.conv import MessagePassing, HeteroConv, GENConv
import torch
from torch.nn import BatchNorm1d, LayerNorm
import torch.nn.functional as F


class MyMessagePassing(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)

        return out

    def message(self, x_j):
        return x_j


class GINEConv(MessagePassing):
    def __init__(self, mlp: torch.nn.Sequential, bond_encoder: torch.nn.Sequential):

        super(GINEConv, self).__init__(aggr="add")

        self.mlp = mlp
        self.eps = torch.nn.Parameter(torch.Tensor([0.]))
        self.bond_encoder = bond_encoder

    def forward(self, x, edge_index, edge_attr):

        edge_embedding = self.bond_encoder(edge_attr) if edge_attr is not None else None
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index,
                                                           x=x,
                                                           edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        m = torch.relu(x_j + edge_attr) if edge_attr is not None else x_j
        return m

    def update(self, aggr_out):
        return aggr_out


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


class DeeperGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_channels, num_tasks, num_layers, dropout, block='res'):
        super(DeeperGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.block = block

        assert block == 'res'  # GraphConv->LN/BN->ReLU->Res

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.node_features_encoder = torch.nn.Linear(in_channels, hidden_channels)
        self.node_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

        for layer in range(self.num_layers):
            gcn = GENConv(hidden_channels, hidden_channels, edge_dim=edge_channels)
            self.gcns.append(gcn)
            self.norms.append(BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index, edge_attr):
        intermediate = []
        h = self.node_features_encoder(x)

        h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_attr)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        intermediate.append(h)

        for layer in range(1, self.num_layers):
            h1 = self.gcns[layer](h, edge_index, edge_attr)
            h2 = self.norms[layer](h1)
            h = F.relu(h2) + h
            h = F.dropout(h, p=self.dropout, training=self.training)
            intermediate.append(h)

        h = self.node_pred_linear(torch.cat(intermediate, dim=1))

        return h


class DeepHeteroGNN(torch.nn.Module):
    def __init__(self, in_shape, hid_dim, num_layers, dropout, share_weight, use_norm, use_res):
        super().__init__()

        self.dropout = dropout
        self.share_weight = share_weight
        self.num_layers = num_layers
        self.use_norm = use_norm
        self.use_res = use_res

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.encoder = torch.nn.ModuleDict({'vals': torch.nn.Linear(in_shape, hid_dim),
                                            'cons': torch.nn.Linear(in_shape, hid_dim),
                                            'obj': torch.nn.Linear(in_shape, hid_dim)})

        for layer in range(num_layers):
            if layer == 0 or not share_weight:
                self.gcns.append(HeteroConv({
                    ('cons', 'to', 'vals'): GENConv(hid_dim, hid_dim, edge_dim=1),
                    ('vals', 'to', 'cons'): GENConv(hid_dim, hid_dim, edge_dim=1),
                    ('vals', 'to', 'obj'): GENConv(hid_dim, hid_dim, edge_dim=1, norm='layer'),
                    ('obj', 'to', 'vals'): GENConv(hid_dim, hid_dim, edge_dim=1),
                    ('cons', 'to', 'obj'): GENConv(hid_dim, hid_dim, edge_dim=1, norm='layer'),
                    ('obj', 'to', 'cons'): GENConv(hid_dim, hid_dim, edge_dim=1),
                },
                    aggr='mean'))
                if use_norm:
                    self.norms.append(torch.nn.ModuleDict({'obj': LayerNorm(hid_dim),
                                                           'vals': BatchNorm1d(hid_dim),
                                                           'cons': BatchNorm1d(hid_dim)}))

        self.pred_vals = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(hid_dim, 2))
        self.pred_cons = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(hid_dim, 1))

    def forward(self, x_dict, edge_index_dict):
        for k in ['cons', 'vals', 'obj']:
            x_dict[k] = torch.relu(self.encoder[k](x_dict[k]))

        hiddens = []
        for i in range(self.num_layers):
            if self.share_weight:
                i = 0

            h1 = x_dict
            h2 = self.gcns[i](x_dict, edge_index_dict)
            hiddens.append((h2['cons'], h2['vals']))
            if self.use_norm:
                h2 = {k: self.norms[i][k](h2[k]) for k in ['cons', 'vals', 'obj']}
            if self.use_res:
                h = {k: (F.relu(h2[k]) + h1[k]) / 2 for k in ['cons', 'vals', 'obj']}
            else:
                h = {k: F.relu(h2[k]) for k in ['cons', 'vals', 'obj']}
            h = {k: F.dropout(h[k], p=self.dropout, training=self.training) for k in ['cons', 'vals', 'obj']}
            x_dict = h

        cons, vals = zip(*hiddens)
        vals = self.pred_vals(torch.stack(vals, dim=0))  # seq * #val * hidden
        cons = self.pred_cons(torch.stack(cons, dim=0))

        return torch.transpose(vals, 0, 1), cons.squeeze().T
