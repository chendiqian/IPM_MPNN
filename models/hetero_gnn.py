from torch_geometric.nn.conv import HeteroConv
import torch
import torch.nn.functional as F

from models.genconv import GENConv, MLP


class TripartiteHeteroGNN(torch.nn.Module):
    def __init__(self, in_shape, pe_dim, hid_dim, num_layers, dropout, share_weight, use_norm, use_res):
        super().__init__()

        self.dropout = dropout
        self.share_weight = share_weight
        self.num_layers = num_layers
        self.use_res = use_res

        self.gcns = torch.nn.ModuleList()

        self.encoder = torch.nn.ModuleDict({'vals': MLP([in_shape, hid_dim, hid_dim], norm='batch'),
                                            'cons': MLP([in_shape, hid_dim, hid_dim], norm='batch'),
                                            'obj': MLP([in_shape, hid_dim, hid_dim], norm='batch')})

        self.pe_encoder = torch.nn.ModuleDict({
            'vals': MLP([pe_dim, hid_dim, hid_dim]),
            'cons': MLP([pe_dim, hid_dim, hid_dim]),
            'obj': MLP([pe_dim, hid_dim, hid_dim])})

        for layer in range(num_layers):
            if layer == 0 or not share_weight:
                self.gcns.append(
                    HeteroConv({
                        ('cons', 'to', 'vals'): GENConv(in_channels=2 * hid_dim,
                                                        out_channels=hid_dim,
                                                        aggr='softmax',
                                                        msg_norm=use_norm,
                                                        learn_msg_scale=use_norm,
                                                        norm='batch' if use_norm else None,
                                                        bias=True,
                                                        edge_dim=1),
                        ('vals', 'to', 'cons'): GENConv(in_channels=2 * hid_dim,
                                                        out_channels=hid_dim,
                                                        aggr='softmax',
                                                        msg_norm=use_norm,
                                                        learn_msg_scale=use_norm,
                                                        norm='batch' if use_norm else None,
                                                        bias=True,
                                                        edge_dim=1),
                        ('vals', 'to', 'obj'): GENConv(in_channels=2 * hid_dim,
                                                       out_channels=hid_dim,
                                                       aggr='softmax',
                                                       msg_norm=use_norm,
                                                       learn_msg_scale=use_norm,
                                                       norm='layer' if use_norm else None,
                                                       bias=True,
                                                       edge_dim=1),
                        ('obj', 'to', 'vals'): GENConv(in_channels=2 * hid_dim,
                                                       out_channels=hid_dim,
                                                       aggr='softmax',
                                                       msg_norm=use_norm,
                                                       learn_msg_scale=use_norm,
                                                       norm='batch' if use_norm else None,
                                                       bias=True,
                                                       edge_dim=1),
                        ('cons', 'to', 'obj'): GENConv(in_channels=2 * hid_dim,
                                                       out_channels=hid_dim,
                                                       aggr='softmax',
                                                       msg_norm=use_norm,
                                                       learn_msg_scale=use_norm,
                                                       norm='layer' if use_norm else None,
                                                       bias=True,
                                                       edge_dim=1),
                        ('obj', 'to', 'cons'): GENConv(in_channels=2 * hid_dim,
                                                       out_channels=hid_dim,
                                                       aggr='softmax',
                                                       msg_norm=use_norm,
                                                       learn_msg_scale=use_norm,
                                                       norm='batch' if use_norm else None,
                                                       bias=True,
                                                       edge_dim=1),
                    },
                        aggr='cat'))

        self.pred_vals = MLP([2 * hid_dim, hid_dim, 1])
        self.pred_cons = MLP([2 * hid_dim, hid_dim, 1])

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for k in ['cons', 'vals', 'obj']:
            x_dict[k] = torch.cat([self.encoder[k](x_dict[k]),
                                   0.5 * (self.pe_encoder[k](data[k].laplacian_eigenvector_pe) +
                                          self.pe_encoder[k](-data[k].laplacian_eigenvector_pe))], dim=1)

        hiddens = []
        for i in range(self.num_layers):
            if self.share_weight:
                i = 0

            h1 = x_dict
            h2 = self.gcns[i](x_dict, edge_index_dict, edge_attr_dict)
            keys = h2.keys()
            hiddens.append((h2['cons'], h2['vals']))
            if self.use_res:
                h = {k: (F.relu(h2[k]) + h1[k]) / 2 for k in keys}
            else:
                h = {k: F.relu(h2[k]) for k in keys}
            h = {k: F.dropout(h[k], p=self.dropout, training=self.training) for k in keys}
            x_dict = h

        cons, vals = zip(*hiddens)
        vals = self.pred_vals(torch.stack(vals, dim=0))  # seq * #val * hidden
        cons = self.pred_cons(torch.stack(cons, dim=0))

        return vals.squeeze().T, cons.squeeze().T


class BipartiteHeteroGNN(torch.nn.Module):
    def __init__(self, in_shape, pe_dim, hid_dim, num_layers, use_norm):
        super().__init__()

        self.num_layers = num_layers

        self.vals2cons = torch.nn.ModuleList()
        self.cons2vals = torch.nn.ModuleList()

        self.encoder = torch.nn.ModuleDict({'vals': torch.nn.Linear(in_shape, hid_dim // 2),
                                            'cons': torch.nn.Linear(in_shape, hid_dim // 2)})

        self.pe_encoder = torch.nn.ModuleDict({
            'vals': torch.nn.Sequential(torch.nn.Linear(pe_dim, hid_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hid_dim, hid_dim // 2)),
            'cons': torch.nn.Sequential(torch.nn.Linear(pe_dim, hid_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hid_dim, hid_dim // 2))})

        for layer in range(num_layers):
            self.vals2cons.append(GENConv(in_channels=hid_dim,
                                          out_channels=hid_dim,
                                          aggr='softmax',
                                          msg_norm=use_norm,
                                          learn_msg_scale=use_norm,
                                          norm='batch' if use_norm else None,
                                          bias=True,
                                          edge_dim=1))
            self.cons2vals.append(GENConv(in_channels=hid_dim,
                                          out_channels=hid_dim,
                                          aggr='softmax',
                                          msg_norm=use_norm,
                                          learn_msg_scale=use_norm,
                                          norm='batch' if use_norm else None,
                                          bias=True,
                                          edge_dim=1))

        self.pred_vals = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(hid_dim, 1))
        self.pred_cons = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(hid_dim, 1))

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        for k in ['cons', 'vals']:
            x_dict[k] = torch.relu(
                torch.cat([self.encoder[k](x_dict[k]),
                           0.5 * (self.pe_encoder[k](data[k].laplacian_eigenvector_pe) +
                                  self.pe_encoder[k](-data[k].laplacian_eigenvector_pe))], dim=1)
            )
        cons, vals = x_dict['cons'], x_dict['vals']

        hiddens = []
        for i in range(self.num_layers):
            cons = self.vals2cons[i]((vals, cons),
                                     data[('vals', 'to', 'cons')].edge_index,
                                     data[('vals', 'to', 'cons')].edge_weight)

            vals = self.cons2vals[i]((cons, vals),
                                     data[('cons', 'to', 'vals')].edge_index,
                                     data[('cons', 'to', 'vals')].edge_weight)

            hiddens.append((cons, vals))

        cons, vals = zip(*hiddens)
        vals = self.pred_vals(torch.stack(vals, dim=0))  # seq * #val * hidden
        cons = self.pred_cons(torch.stack(cons, dim=0))

        return vals.squeeze().T, cons.squeeze().T
