from typing import List, Optional, Tuple, Union

import torch
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.norm import MessageNorm
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import is_torch_sparse_tensor, to_edge_index
from models.utils import MLP, scatter_sum


# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GENConv.html?highlight=GENConv
class GENConv(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = 'softmax',
        t: float = 1.0,
        learn_t: bool = False,
        p: float = 1.0,
        learn_p: bool = False,
        msg_norm: bool = False,
        learn_msg_scale: bool = False,
        norm: str = 'batch',
        num_layers: int = 2,
        expansion: int = 2,
        eps: float = 1e-7,
        bias: bool = False,
        edge_dim: Optional[int] = None,
        in_place: bool = True,
        **kwargs,
    ):

        # Backward compatibility:
        semi_grad = True if aggr == 'softmax_sg' else False
        aggr = 'softmax' if aggr == 'softmax_sg' else aggr
        aggr = 'powermean' if aggr == 'power' else aggr

        # Override args of aggregator if `aggr_kwargs` is specified
        if 'aggr_kwargs' not in kwargs:
            if aggr == 'softmax':
                kwargs['aggr_kwargs'] = dict(t=t, learn=learn_t,
                                             semi_grad=semi_grad)
            elif aggr == 'powermean':
                kwargs['aggr_kwargs'] = dict(p=p, learn=learn_p)

        self.in_place = in_place

        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = eps

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if in_channels[0] != out_channels:
            self.lin_src = Linear(in_channels[0], out_channels, bias=bias)

        if edge_dim is not None and edge_dim != out_channels:
            # self.lin_edge = Linear(edge_dim, out_channels, bias=bias)
            self.lin_edge = MLP([edge_dim, out_channels, out_channels], norm='batch', bias=bias)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(out_channels)
        else:
            aggr_out_channels = out_channels

        if aggr_out_channels != out_channels:
            self.lin_aggr_out = Linear(aggr_out_channels, out_channels,
                                       bias=bias)

        if in_channels[1] != out_channels:
            self.lin_dst = Linear(in_channels[1], out_channels, bias=bias)

        channels = [out_channels]
        for i in range(num_layers - 1):
            channels.append(out_channels * expansion)
        channels.append(out_channels)
        self.mlp = MLP(channels, norm=norm, bias=bias)

        if msg_norm:
            self.msg_norm = MessageNorm(learn_msg_scale)

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp)
        if hasattr(self, 'msg_norm'):
            self.msg_norm.reset_parameters()
        if hasattr(self, 'lin_src'):
            self.lin_src.reset_parameters()
        if hasattr(self, 'lin_edge'):
            reset(self.lin_edge)
        if hasattr(self, 'lin_aggr_out'):
            self.lin_aggr_out.reset_parameters()
        if hasattr(self, 'lin_dst'):
            self.lin_dst.reset_parameters()

    def forward(self, x: Union[torch.Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> torch.Tensor:

        if isinstance(x, torch.Tensor):
            x: OptPairTensor = (x, x)

        if hasattr(self, 'lin_src'):
            x = (self.lin_src(x[0]), x[1])

        if isinstance(edge_index, SparseTensor):
            edge_attr = edge_index.storage.value()
        elif is_torch_sparse_tensor(edge_index):
            _, value = to_edge_index(edge_index)
            if value.dim() > 1 or not value.all():
                edge_attr = value

        if edge_attr is not None and hasattr(self, 'lin_edge'):
            edge_attr = self.lin_edge(edge_attr)

        # Node and edge feature dimensionalites need to match.
        if edge_attr is not None:
            assert x[0].size(-1) == edge_attr.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        if self.in_place:
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        else:
            # todo: add softmax non in-place
            msg = x[0][edge_index[0]] if edge_attr is None else x[0][edge_index[0]] + edge_attr
            msg = torch.relu(msg) + self.eps
            out = scatter_sum(msg, edge_index[1], dim=0)

        if hasattr(self, 'lin_aggr_out'):
            out = self.lin_aggr_out(out)

        if hasattr(self, 'msg_norm'):
            h = x[1] if x[1] is not None else x[0]
            assert h is not None
            out = self.msg_norm(h, out)

        x_dst = x[1]
        if x_dst is not None:
            if hasattr(self, 'lin_dst'):
                x_dst = self.lin_dst(x_dst)
            out = out + x_dst

        return self.mlp(out)

    def message(self, x_j: torch.Tensor, edge_attr: OptTensor) -> torch.Tensor:
        msg = x_j if edge_attr is None else x_j + edge_attr
        return msg.relu() + self.eps

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')
