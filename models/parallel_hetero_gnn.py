from torch_geometric.nn.conv import HeteroConv, GENConv
import torch
import torch.nn.functional as F


class ParallelHeteroGNN(torch.nn.Module):
    def __init__(self, bipartite, in_shape, pe_dim, hid_dim, num_layers, dropout, share_weight, use_norm, use_res):
        super().__init__()

        self.dropout = dropout
        self.share_weight = share_weight
        self.num_layers = num_layers
        self.use_res = use_res

        self.gcns = torch.nn.ModuleList()

        self.encoder = torch.nn.ModuleDict({'vals': torch.nn.Linear(in_shape, hid_dim // 2),
                                            'cons': torch.nn.Linear(in_shape, hid_dim // 2),
                                            'obj': torch.nn.Linear(in_shape, hid_dim // 2)})

        self.pe_encoder = torch.nn.ModuleDict({'vals': torch.nn.Linear(pe_dim, hid_dim // 2),
                                            'cons': torch.nn.Linear(pe_dim, hid_dim // 2),
                                            'obj': torch.nn.Linear(pe_dim, hid_dim // 2)})

        for layer in range(num_layers):
            if layer == 0 or not share_weight:
                self.gcns.append(
                    HeteroConv({
                        ('cons', 'to', 'vals'): GENConv(in_channels=hid_dim,
                                                        out_channels=hid_dim,
                                                        aggr='softmax',
                                                        msg_norm=use_norm,
                                                        learn_msg_scale=use_norm,
                                                        norm='batch' if use_norm else None,
                                                        bias=True,
                                                        edge_dim=1),
                        ('vals', 'to', 'cons'): GENConv(in_channels=hid_dim,
                                                        out_channels=hid_dim,
                                                        aggr='softmax',
                                                        msg_norm=use_norm,
                                                        learn_msg_scale=use_norm,
                                                        norm='batch' if use_norm else None,
                                                        bias=True,
                                                        edge_dim=1)
                    },
                        aggr='mean') if bipartite else
                    HeteroConv({
                        ('cons', 'to', 'vals'): GENConv(in_channels=hid_dim,
                                                        out_channels=hid_dim,
                                                        aggr='softmax',
                                                        msg_norm=use_norm,
                                                        learn_msg_scale=use_norm,
                                                        norm='batch' if use_norm else None,
                                                        bias=True,
                                                        edge_dim=1),
                        ('vals', 'to', 'cons'): GENConv(in_channels=hid_dim,
                                                        out_channels=hid_dim,
                                                        aggr='softmax',
                                                        msg_norm=use_norm,
                                                        learn_msg_scale=use_norm,
                                                        norm='batch' if use_norm else None,
                                                        bias=True,
                                                        edge_dim=1),
                        ('vals', 'to', 'obj'): GENConv(in_channels=hid_dim,
                                                       out_channels=hid_dim,
                                                       aggr='softmax',
                                                       msg_norm=use_norm,
                                                       learn_msg_scale=use_norm,
                                                       norm='layer' if use_norm else None,
                                                       bias=True,
                                                       edge_dim=1),
                        ('obj', 'to', 'vals'): GENConv(in_channels=hid_dim,
                                                       out_channels=hid_dim,
                                                       aggr='softmax',
                                                       msg_norm=use_norm,
                                                       learn_msg_scale=use_norm,
                                                       norm='batch' if use_norm else None,
                                                       bias=True,
                                                       edge_dim=1),
                        ('cons', 'to', 'obj'): GENConv(in_channels=hid_dim,
                                                       out_channels=hid_dim,
                                                       aggr='softmax',
                                                       msg_norm=use_norm,
                                                       learn_msg_scale=use_norm,
                                                       norm='layer' if use_norm else None,
                                                       bias=True,
                                                       edge_dim=1),
                        ('obj', 'to', 'cons'): GENConv(in_channels=hid_dim,
                                                       out_channels=hid_dim,
                                                       aggr='softmax',
                                                       msg_norm=use_norm,
                                                       learn_msg_scale=use_norm,
                                                       norm='batch' if use_norm else None,
                                                       bias=True,
                                                       edge_dim=1),
                    },
                        aggr='mean'))

        self.pred_vals = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(hid_dim, 2))
        self.pred_cons = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(hid_dim, 1))

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        for k in ['cons', 'vals', 'obj']:
            x_dict[k] = torch.relu(
                torch.cat([self.encoder[k](x_dict[k]),
                           0.5 * (self.pe_encoder[k](data[k].laplacian_eigenvector_pe) +
                                  self.pe_encoder[k](-data[k].laplacian_eigenvector_pe))], dim=1)
            )

        hiddens = []
        for i in range(self.num_layers):
            if self.share_weight:
                i = 0

            h1 = x_dict
            h2 = self.gcns[i](x_dict, edge_index_dict)
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

        return torch.transpose(vals, 0, 1), cons.squeeze().T
