from typing import Dict, List
import math

import torch
import numpy as np
from torch_geometric.data import Data, Batch


def log_normalize(x):
    return torch.log(1. + x)


def log_denormalize(x):
    return torch.exp(x) - 1.


def mode_of_distribution(x):
    cnt, intervals = np.histogram(x, bins=50, range=None, density=None, weights=None)
    idx = cnt.argmax()
    return (intervals[idx] + intervals[idx + 1]) / 2


def args_set_bool(args: Dict):
    for k, v in args.items():
        if isinstance(v, str):
            if v.lower() == 'true':
                args[k] = True
            elif v.lower() == 'false':
                args[k] = False
    return args


def barrier_function(x, t=1.e5):
    cond = x.detach() >= 1 / (t ** 2)
    return torch.where(cond, (-1 / t) * torch.log(x), -t * x - 1 / t * math.log(1 / (t ** 2)) + 1 / t)


def collate_fn_with_repeats(graphs: List[Data], repeats: int):
    # [1, 1, 1, 2, 2, 2, 3, 3, 3]
    num_graphs = len(graphs)
    original_batch = collate_fn_ip(graphs)
    num_val_nodes = torch.tensor([g['vals'].x.shape[0] for g in graphs])
    num_con_nodes = torch.tensor([g['cons'].x.shape[0] for g in graphs])
    original_batch.num_val_nodes = num_val_nodes
    original_batch.num_con_nodes = num_con_nodes
    original_batch.repeats = 1

    cumsum_vals = torch.cat([num_val_nodes.new_zeros(1, dtype=torch.long),
                             torch.cumsum(torch.vstack([num_val_nodes, num_con_nodes]).sum(0), dim=0)], dim=0)
    batch_val = torch.cat([torch.arange(n).repeat(repeats) + cumsum_vals[i] for i, n in enumerate(num_val_nodes)],
                          dim=0)
    cumsum_cons = torch.cumsum(torch.vstack([torch.hstack([num_val_nodes, torch.zeros(1, dtype=torch.long)]),
                                             torch.hstack([torch.zeros(1, dtype=torch.long), num_con_nodes])]).sum(0),
                               dim=0)
    batch_con = torch.cat([torch.arange(n).repeat(repeats) + cumsum_cons[i] for i, n in enumerate(num_con_nodes)],
                          dim=0)
    graphs = [g for g in graphs for _ in range(repeats)]
    batch = collate_fn_ip(graphs)
    batch.num_val_nodes = num_val_nodes
    batch.num_con_nodes = num_con_nodes
    # batch.num_graphs = num_graphs
    batch.repeats = repeats
    batch.val_con_batch = torch.cat([batch_val, batch_con], dim=0)
    return batch, original_batch


def collate_fn_ip(graphs: List[Data]):
    new_batch = Batch.from_data_list(graphs)
    row_bias = torch.hstack([new_batch.A_num_row.new_zeros(1), new_batch.A_num_row[:-1]]).cumsum(dim=0)
    row_bias = torch.repeat_interleave(row_bias, new_batch.A_nnz)
    new_batch.A_row += row_bias
    col_bias = torch.hstack([new_batch.A_num_col.new_zeros(1), new_batch.A_num_col[:-1]]).cumsum(dim=0)
    col_bias = torch.repeat_interleave(col_bias, new_batch.A_nnz)
    new_batch.A_col += col_bias
    return new_batch
