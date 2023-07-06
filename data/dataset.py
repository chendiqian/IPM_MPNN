import gzip
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, Batch, HeteroData, InMemoryDataset
from torch_sparse import SparseTensor

from scipy_solver.linprog import linprog


def collate_fn_ip(graphs: List[Data]):
    new_batch = Batch.from_data_list(graphs)
    row_bias = torch.hstack([new_batch.A_num_row.new_zeros(1), new_batch.A_num_row[:-1]]).cumsum(dim=0)
    row_bias = torch.repeat_interleave(row_bias, new_batch.A_nnz)
    new_batch.A_row += row_bias
    col_bias = torch.hstack([new_batch.A_num_col.new_zeros(1), new_batch.A_num_col[:-1]]).cumsum(dim=0)
    col_bias = torch.repeat_interleave(col_bias, new_batch.A_nnz)
    new_batch.A_col += col_bias
    return new_batch


class SetCoverDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        normalize: bool,
        rand_starts: int = 10,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.rand_starts = rand_starts
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, 'data.pt')
        self.data, self.slices = torch.load(path)

        if normalize:
            self.std = self.data.gt_primals.std()
            self.mean = self.data.gt_primals.mean()
            self.data.gt_primals = (self.data.gt_primals - self.mean) / self.std
            for k in ['cons', 'vals']:
                self.data[k].x = (self.data[k].x -
                                       self.data[k].x.mean(0, keepdims=True)) / \
                                      self.data[k].x.std(0, keepdims=True)
        else:
            self.std, self.mean = 1., 0.


    @property
    def raw_file_names(self) -> List[str]:
        return ['instance_0.pkl.gz']   # there should be at least one pkg

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'processed_{self.rand_starts}restarts')

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def process(self):
        num_instance_pkg = len([n for n in os.listdir(self.raw_dir) if n.endswith('pkl.gz')])

        data_list = []
        for i in range(num_instance_pkg):
            # load instance
            print(f"processing {i}th package, {num_instance_pkg} in total")
            with gzip.open(os.path.join(self.raw_dir, f"instance_{i}.pkl.gz"), "rb") as file:
                ip_pkgs = pickle.load(file)

            for ip_idx in range(len(ip_pkgs)):
                (A, b, c) = ip_pkgs[ip_idx]
                A_tilde = A.clone()
                A_tilde[:, -A.shape[0]:] = 0.
                sp_mat = SparseTensor.from_dense(A_tilde, has_value=True)

                row = sp_mat.storage._row
                col = sp_mat.storage._col
                val = sp_mat.storage._value

                c = c / c.max()  # does not change the result

                for _ in range(self.rand_starts):
                    # solve the LP
                    # sol = ipm_overleaf(c.numpy(), None, None, A.numpy(), b.numpy(), None, max_iter=1000, lin_solver='scipy_cg')

                    sol = linprog(c.numpy(),
                                  A_ub=None,
                                  b_ub=None,
                                  A_eq=A.numpy(), b_eq=b.numpy(), bounds=None,
                                  method='interior-point', callback=lambda res: res.x)

                    # organize results
                    # x, l, s = zip(*sol['xs'])
                    # x = np.stack(x, axis=1)  # primal
                    # l = np.stack(l, axis=1)  # dual
                    # s = np.stack(s, axis=1)  # slack

                    # x = np.stack([i['x'] for i in sol.intermediate], axis=1)
                    x = np.stack(sol.intermediate, axis=1)

                    # l = np.stack([i['con'] for i in sol.intermediate], axis=1)

                    gt_primals = torch.from_numpy(x).to(torch.float)
                    # gt_duals = torch.from_numpy(l).to(torch.float)
                    # gt_slacks = torch.from_numpy(s).to(torch.float)

                    data = HeteroData(
                        cons={'x': torch.cat([A.mean(1, keepdims=True),
                                              A.std(1, keepdims=True)], dim=1)},
                        vals={'x': torch.cat([A.mean(0, keepdims=True),
                                              A.std(0, keepdims=True)], dim=0).T},
                        obj={'x': torch.cat([c.mean(0, keepdims=True),
                                             c.std(0, keepdims=True)], dim=0)[None]},

                        cons__to__vals={'edge_index': torch.vstack(torch.where(A)),
                                        'edge_attr': A[torch.where(A)][:, None]},
                        vals__to__cons={'edge_index': torch.vstack(torch.where(A.T)),
                                        'edge_attr': A.T[torch.where(A.T)][:, None]},
                        vals__to__obj={'edge_index': torch.vstack([torch.arange(A.shape[1]),
                                                                   torch.zeros(A.shape[1], dtype=torch.long)]),
                                       'edge_attr': c[:, None]},
                        obj__to__vals={'edge_index': torch.vstack([torch.zeros(A.shape[1], dtype=torch.long),
                                                                   torch.arange(A.shape[1])]),
                                       'edge_attr': c[:, None]},
                        cons__to__obj={'edge_index': torch.vstack([torch.arange(A.shape[0]),
                                                                   torch.zeros(A.shape[0], dtype=torch.long)]),
                                       'edge_attr': b[:, None]},
                        obj__to__cons={'edge_index': torch.vstack([torch.zeros(A.shape[0], dtype=torch.long),
                                                                   torch.arange(A.shape[0])]),
                                       'edge_attr': b[:, None]},
                        gt_primals=gt_primals,
                        # gt_duals=gt_duals,
                        # gt_slacks=gt_slacks,
                        obj_value=torch.tensor(sol['fun'].astype(np.float32)),
                        obj_const=c,

                        A_row=row,
                        A_col=col,
                        A_val=val,
                        A_num_row=A_tilde.shape[0],
                        A_num_col=A_tilde.shape[1],
                        A_nnz=len(val),
                        rhs=b)

                    if self.pre_filter is not None:
                        raise NotImplementedError

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)

        torch.save(self.collate(data_list), osp.join(self.processed_dir, 'data.pt'))
