import torch


def log_normalize(x):
    return torch.log(1. + x)


def log_denormalize(x):
    return torch.exp(x) - 1.
