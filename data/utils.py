from typing import Dict
import math

import torch
import numpy as np


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
