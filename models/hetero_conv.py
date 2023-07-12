import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torch.nn import ModuleList
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import Adj, EdgeType, NodeType
from torch_geometric.utils.hetero import check_add_self_loops


def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    elif aggr == "cat":
        return torch.cat(xs, dim=-1)
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out


class HeteroConv(torch.nn.Module):
    def __init__(
        self,
        convs: Dict[EdgeType, Tuple[MessagePassing, int]],
        aggr: Optional[str] = "sum",
    ):
        super().__init__()

        for edge_type, (module, _) in convs.items():
            check_add_self_loops(module, [edge_type])

        src_node_types = set([key[0] for key in convs.keys()])
        dst_node_types = set([key[-1] for key in convs.keys()])
        if len(src_node_types - dst_node_types) > 0:
            warnings.warn(
                f"There exist node types ({src_node_types - dst_node_types}) "
                f"whose representations do not get updated during message "
                f"passing as they do not occur as destination type in any "
                f"edge type. This may lead to unexpected behavior.")

        self.convs = ModuleDict({'__'.join(k): v[0] for k, v in convs.items()})
        conv_rank = {'__'.join(k): v[1] for k, v in convs.items()}
        sorted_rank_value = sorted(set(conv_rank.values()))  # small value has high priority

        self.ranked_convs = ModuleList([])
        for i, rank in enumerate(sorted_rank_value):
            module_dict = ModuleDict({})
            for k, v in conv_rank.items():
                if v == rank:
                    module_dict[k] = self.convs[k]
            self.ranked_convs.append(module_dict)

        self.aggr = aggr

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        # for conv in self.convs.values():
        #     conv.reset_parameters()
        raise NotImplementedError

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        *args_dict,
        **kwargs_dict,
    ) -> Dict[NodeType, Tensor]:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :class:`torch.Tensor` of
                shape :obj:`[2, num_edges]` or a
                :class:`torch_sparse.SparseTensor`.
            *args_dict (optional): Additional forward arguments of invididual
                :class:`torch_geometric.nn.conv.MessagePassing` layers.
            **kwargs_dict (optional): Additional forward arguments of
                individual :class:`torch_geometric.nn.conv.MessagePassing`
                layers.
                For example, if a specific GNN layer at edge type
                :obj:`edge_type` expects edge attributes :obj:`edge_attr` as a
                forward argument, then you can pass them to
                :meth:`~torch_geometric.nn.conv.HeteroConv.forward` via
                :obj:`edge_attr_dict = { edge_type: edge_attr }`.
        """
        for cur_rank, cur_convs in enumerate(self.ranked_convs):
            out_dict = defaultdict(list)
            for edge_type, edge_index in edge_index_dict.items():
                src, rel, dst = edge_type

                str_edge_type = '__'.join(edge_type)
                if str_edge_type not in cur_convs:
                    continue

                args = []
                for value_dict in args_dict:
                    if edge_type in value_dict:
                        args.append(value_dict[edge_type])
                    elif src == dst and src in value_dict:
                        args.append(value_dict[src])
                    elif src in value_dict or dst in value_dict:
                        args.append(
                            (value_dict.get(src, None), value_dict.get(dst, None)))

                kwargs = {}
                for arg, value_dict in kwargs_dict.items():
                    arg = arg[:-5]  # `{*}_dict`
                    if edge_type in value_dict:
                        kwargs[arg] = value_dict[edge_type]
                    elif src == dst and src in value_dict:
                        kwargs[arg] = value_dict[src]
                    elif src in value_dict or dst in value_dict:
                        kwargs[arg] = (value_dict.get(src, None),
                                       value_dict.get(dst, None))

                conv = cur_convs[str_edge_type]

                if src == dst:
                    out = conv(x_dict[src], edge_index, *args, **kwargs)
                else:
                    out = conv((x_dict[src], x_dict[dst]), edge_index, *args,
                               **kwargs)

                out_dict[dst].append(out)

            for key, value in out_dict.items():
                x_dict[key] = group(value, self.aggr)

        return x_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'
