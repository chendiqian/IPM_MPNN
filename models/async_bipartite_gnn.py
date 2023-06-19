import torch
from torch_geometric.nn.conv import GENConv


class UnParallelHeteroGNN(torch.nn.Module):
    def __init__(self, in_shape, pe_dim, hid_dim, num_layers, use_norm):
        super().__init__()

        self.num_layers = num_layers

        self.vals2cons = torch.nn.ModuleList()
        self.cons2vals = torch.nn.ModuleList()

        self.encoder = torch.nn.ModuleDict({'vals': torch.nn.Linear(in_shape, hid_dim // 2),
                                            'cons': torch.nn.Linear(in_shape, hid_dim // 2)})

        self.pe_encoder = torch.nn.ModuleDict({'vals': torch.nn.Linear(pe_dim, hid_dim // 2),
                                               'cons': torch.nn.Linear(pe_dim, hid_dim // 2)})

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
                                             torch.nn.Linear(hid_dim, 2))
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

        return torch.transpose(vals, 0, 1), cons.squeeze().T
