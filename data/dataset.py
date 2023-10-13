import gzip
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData, InMemoryDataset
from torch_sparse import SparseTensor

from solver.linprog import linprog
from tqdm import tqdm


class LPDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        extra_path: str,
        upper_bound: Optional = None,
        rand_starts: int = 1,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.rand_starts = rand_starts
        self.using_ineq = True
        self.extra_path = extra_path
        self.upper_bound = upper_bound
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, 'data.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return ['instance_0.pkl.gz']   # there should be at least one pkg

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed_' + self.extra_path)

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

            for ip_idx in tqdm(range(len(ip_pkgs))):
                (A, b, c) = ip_pkgs[ip_idx]
                sp_a = SparseTensor.from_dense(A, has_value=True)

                row = sp_a.storage._row
                col = sp_a.storage._col
                val = sp_a.storage._value

                if self.using_ineq:
                    tilde_mask = torch.ones(row.shape, dtype=torch.bool)
                else:
                    tilde_mask = col < (A.shape[1] - A.shape[0])

                c = c / (c.abs().max() + 1.e-10)  # does not change the result

                # solve the LP
                if self.using_ineq:
                    A_ub = A.numpy()
                    b_ub = b.numpy()
                    A_eq = None
                    b_eq = None
                else:
                    A_eq = A.numpy()
                    b_eq = b.numpy()
                    A_ub = None
                    b_ub = None

                bounds = (0, self.upper_bound)

                for _ in range(self.rand_starts):
                    # sol = ipm_overleaf(c.numpy(), A_ub, b_ub, A_eq, b_eq, None, max_iter=1000, lin_solver='scipy_cg')
                    # x = np.stack(sol['xs'], axis=1)  # primal

                    sol = linprog(c.numpy(),
                                  A_ub=A_ub,
                                  b_ub=b_ub,
                                  A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                                  method='interior-point', callback=lambda res: res.x)
                    x = np.stack(sol.intermediate, axis=1)
                    assert not np.isnan(sol['fun'])

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
                        A_num_row=A.shape[0],
                        A_num_col=A.shape[1],
                        A_nnz=len(val),
                        A_tilde_mask=tilde_mask,
                        rhs=b)

                    if self.pre_filter is not None:
                        raise NotImplementedError

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)

            torch.save(Batch.from_data_list(data_list), osp.join(self.processed_dir, f'batch{i}.pt'))
            data_list = []

        data_list = []
        for i in range(num_instance_pkg):
            data_list.extend(Batch.to_data_list(torch.load(osp.join(self.processed_dir, f'batch{i}.pt'))))
        torch.save(self.collate(data_list), osp.join(self.processed_dir, 'data.pt'))
