from models.hetero_gnn import TripartiteHeteroGNN
import torch
import torch.nn.functional as F

from models.utils import MLP


class TimeDependentTripartiteHeteroGNN(TripartiteHeteroGNN):
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
                 use_norm,
                 use_res,
                 conv_sequence='cov'):
        # torch.manual_seed(42)
        # torch.cuda.manual_seed(42)
        super().__init__(
            conv,
            in_shape,
            pe_dim,
            hid_dim,
            num_conv_layers,
            num_pred_layers,
            num_mlp_layers,
            dropout,
            share_conv_weight,
            True,
            use_norm,
            use_res,
            False,
            conv_sequence)
        in_emb_dim = hid_dim if pe_dim > 0 else 2 * hid_dim
        self.time_encoder = MLP([1, hid_dim, in_emb_dim])
        # self.encoder = torch.nn.ModuleDict({'vals': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
        #                                     'cons': MLP([in_shape, hid_dim, in_emb_dim], norm='batch'),
        #                                     'obj': MLP([in_shape, hid_dim, in_emb_dim], norm='batch')})

    def forward(self, t, data):
        # t: shape (N,)
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for k in ['cons', 'vals', 'obj']:
            x_emb = self.encoder[k](x_dict[k]) + (self.time_encoder(t.reshape(-1, 1)) if k == 'obj' else 0.)
            if self.pe_encoder is not None and hasattr(data[k], 'laplacian_eigenvector_pe'):
                pe_emb = 0.5 * (self.pe_encoder[k](data[k].laplacian_eigenvector_pe) +
                                self.pe_encoder[k](-data[k].laplacian_eigenvector_pe))
                x_emb = torch.cat([x_emb, pe_emb], dim=1)
            x_dict[k] = x_emb

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

        cons, vals = hiddens[-1]

        vals = self.pred_vals(vals).squeeze()  #val * hidden -> #val
        cons = self.pred_cons(cons).squeeze()

        val_con = torch.cat([vals, cons], dim=0)
        val_con = val_con * (1 - torch.exp(-t))

        return val_con
