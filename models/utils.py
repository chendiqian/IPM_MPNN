from typing import Optional, List

import torch
from torch.nn import Sequential, BatchNorm1d, LayerNorm, InstanceNorm1d, ReLU, Dropout
from torch_geometric.nn import Linear
from torch_scatter.utils import broadcast


# originally this was in-place, for torch.vmap(torch.func.jacrec) reasons we need non in-place version
# https://pytorch-scatter.readthedocs.io/en/latest/_modules/torch_scatter/scatter.html#scatter
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return torch.scatter_add(out, dim, index, src)
    else:
        return torch.scatter_add(out, dim, index, src)


class MLP(Sequential):
    def __init__(self, channels: List[int], norm: Optional[str] = None,
                 bias: bool = True, dropout: float = 0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias=bias))

            if i < len(channels) - 1:
                if norm and norm == 'batch':
                    m.append(BatchNorm1d(channels[i], affine=True))
                elif norm and norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                elif norm and norm == 'instance':
                    m.append(InstanceNorm1d(channels[i], affine=False))
                elif norm:
                    raise NotImplementedError(
                        f'Normalization layer "{norm}" not supported.')
                m.append(ReLU())
                m.append(Dropout(dropout))

        super().__init__(*m)
