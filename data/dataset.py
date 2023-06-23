import gzip
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from scipy_solver.linprog import linprog
from solver import ipm_overleaf


class SetCoverDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, 'data.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return [f'instance_{i}.pkl.gz' for i in range(1000)]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def process(self):
        num_instance = len(os.listdir(self.raw_dir))

        data_list = []
        for i in range(num_instance):
            # load instance
            print(f"processing {i}th instance")
            with gzip.open(os.path.join(self.raw_dir, f"instance_{i}.pkl.gz"), "rb") as file:
                (A, b, c) = pickle.load(file)

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

            x = np.stack(sol.intermediate, axis=1)

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
                                'edge_weight': A[torch.where(A)][:, None]},
                vals__to__cons={'edge_index': torch.vstack(torch.where(A.T)),
                                'edge_weight': A.T[torch.where(A.T)][:, None]},
                vals__to__obj={'edge_index': torch.vstack(
                    [torch.arange(A.shape[1]), torch.zeros(A.shape[1], dtype=torch.long)]),
                               'edge_weight': torch.nn.functional.normalize(c, p=2.0, dim=0)[
                                              :, None]},
                obj__to__vals={'edge_index': torch.vstack(
                    [torch.zeros(A.shape[1], dtype=torch.long), torch.arange(A.shape[1])]),
                               'edge_weight': torch.nn.functional.normalize(c, p=2.0, dim=0)[
                                              :, None]},
                cons__to__obj={'edge_index': torch.vstack(
                    [torch.arange(A.shape[0]), torch.zeros(A.shape[0], dtype=torch.long)]),
                               'edge_weight': torch.nn.functional.normalize(b, p=2.0, dim=0)[
                                              :, None]},
                obj__to__cons={'edge_index': torch.vstack(
                    [torch.zeros(A.shape[0], dtype=torch.long), torch.arange(A.shape[0])]),
                               'edge_weight': torch.nn.functional.normalize(b, p=2.0, dim=0)[
                                              :, None]},
                gt_primals=gt_primals,
                # gt_duals=gt_duals,
                # gt_slacks=gt_slacks,
                obj_value=sol['fun'],
                obj_const=c)


            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), osp.join(self.processed_dir, 'data.pt'))
