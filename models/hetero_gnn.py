import torch
import torch.nn.functional as F

from models.genconv import GENConv
from models.gcnconv import GCNConv
from models.utils import MLP
from models.hetero_conv import HeteroConv


def strseq2rank(conv_sequence):
    if conv_sequence == 'parallel':
        c2v = v2c = v2o = o2v = c2o = o2c = 0
    elif conv_sequence == 'cvo':
        v2c = o2c = 0
        c2v = o2v = 1
        c2o = v2o = 2
    elif conv_sequence == 'vco':
        c2v = o2v = 0
        v2c = o2c = 1
        c2o = v2o = 2
    elif conv_sequence == 'ocv':
        c2o = v2o = 0
        v2c = o2c = 1
        c2v = o2v = 2
    elif conv_sequence == 'ovc':
        c2o = v2o = 0
        c2v = o2v = 1
        v2c = o2c = 2
    elif conv_sequence == 'voc':
        c2v = o2v = 0
        c2o = v2o = 1
        v2c = o2c = 2
    elif conv_sequence == 'cov':
        v2c = o2c = 0
        c2o = v2o = 1
        c2v = o2v = 2
    else:
        raise ValueError
    return c2v, v2c, v2o, o2v, c2o, o2c


def get_conv_layer(conv: str,
             hid_dim: int,
             num_mlp_layers: int,
             use_norm: bool,
             in_place: bool):
    if conv.lower() == 'genconv':
        def get_conv():
            return GENConv(in_channels=2 * hid_dim,
                    out_channels=hid_dim,
                    num_layers=num_mlp_layers,
                    aggr='softmax',
                    msg_norm=use_norm,
                    learn_msg_scale=use_norm,
                    norm='batch' if use_norm else None,
                    bias=True,
                    edge_dim=1,
                    in_place=in_place)
    elif conv.lower() == 'gcnconv':
        def get_conv():
            return GCNConv(in_dim=2 * hid_dim,
                           edge_dim=1,
                           hid_dim=hid_dim,
                           num_mlp_layers=num_mlp_layers,
                           norm='batch' if use_norm else None,
                           in_place=in_place)
    else:
        raise NotImplementedError

    return get_conv


class TripartiteHeteroGNN(torch.nn.Module):
    def __init__(self,
                 conv,
                 in_shape,
                 pe_dim,
                 hid_dim,
                 num_conv_layers,
                 num_pred_layers,
                 num_mlp_layers,
                 dropout,
                 share_conv_weight,
                 share_lin_weight,
                 use_norm,
                 use_res,
                 in_place=True,
                 conv_sequence='parallel'):
        super().__init__()

        self.dropout = dropout
        self.share_conv_weight = share_conv_weight
        self.share_lin_weight = share_lin_weight
        self.num_layers = num_conv_layers
        self.use_res = use_res

        self.encoder = torch.nn.ModuleDict({'vals': MLP([in_shape, hid_dim, hid_dim], norm='batch'),
                                            'cons': MLP([in_shape, hid_dim, hid_dim], norm='batch'),
                                            'obj': MLP([in_shape, hid_dim, hid_dim], norm='batch')})

        self.pe_encoder = torch.nn.ModuleDict({
            'vals': MLP([pe_dim, hid_dim, hid_dim]),
            'cons': MLP([pe_dim, hid_dim, hid_dim]),
            'obj': MLP([pe_dim, hid_dim, hid_dim])})

        c2v, v2c, v2o, o2v, c2o, o2c = strseq2rank(conv_sequence)
        get_conv = get_conv_layer(conv, hid_dim, num_mlp_layers, use_norm, in_place)
        self.gcns = torch.nn.ModuleList()
        for layer in range(num_conv_layers):
            if layer == 0 or not share_conv_weight:
                self.gcns.append(
                    HeteroConv({
                        ('cons', 'to', 'vals'): (get_conv(), c2v),
                        ('vals', 'to', 'cons'): (get_conv(), v2c),
                        ('vals', 'to', 'obj'): (get_conv(), v2o),
                        ('obj', 'to', 'vals'): (get_conv(), o2v),
                        ('cons', 'to', 'obj'): (get_conv(), c2o),
                        ('obj', 'to', 'cons'): (get_conv(), o2c),
                    }, aggr='cat'))

        if share_lin_weight:
            self.pred_vals = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])
            self.pred_cons = MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1])
        else:
            self.pred_vals = torch.nn.ModuleList()
            self.pred_cons = torch.nn.ModuleList()
            for layer in range(num_conv_layers):
                self.pred_vals.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))
                self.pred_cons.append(MLP([2 * hid_dim] + [hid_dim] * (num_pred_layers - 1) + [1]))

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for k in ['cons', 'vals', 'obj']:
            x_dict[k] = torch.cat([self.encoder[k](x_dict[k]),
                                   0.5 * (self.pe_encoder[k](data[k].laplacian_eigenvector_pe) +
                                          self.pe_encoder[k](-data[k].laplacian_eigenvector_pe))], dim=1)

        hiddens = []
        for i in range(self.num_layers):
            if self.share_conv_weight:
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

        if self.share_lin_weight:
            vals = self.pred_vals(torch.stack(vals, dim=0))  # seq * #val * hidden
            cons = self.pred_cons(torch.stack(cons, dim=0))
            return vals.squeeze().T, cons.squeeze().T
        else:
            vals = torch.cat([self.pred_vals[i](vals[i]) for i in range(self.num_layers)], dim=1)
            cons = torch.cat([self.pred_cons[i](cons[i]) for i in range(self.num_layers)], dim=1)
            return vals, cons


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
