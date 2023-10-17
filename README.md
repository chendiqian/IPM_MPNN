# IPM-GNN: Exploring the Power of Graph Neural Networks in Solving Linear Optimization Problems

[![arXiv](https://img.shields.io/badge/arXiv-2310.10603-b31b1b.svg)](https://arxiv.org/abs/2310.10603)

<img src="https://github.com/chendiqian/IPM_MPNN/blob/master/overview.jpg" alt="drawing" width="900"/>
<p align="center">
An overview of our IPM-MPNN
</p>



# Environment setup

Simply install from the existing file `conda env create -f environment.yml` or install the required envs manually:

```angular2html
conda create -y -n ipmgnn python=3.10
conda activate ipmgnn
conda install pytorch==2.0.0  pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_scatter-2.1.1%2Bpt20cu118-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_sparse-0.6.17%2Bpt20cu118-cp310-cp310-linux_x86_64.whl
pip install ml-collections
pip install wandb
```

# Reproduction of the results

## Main results 

See `run/main.sh` for the commands and hyperparameters. Note that the small and large instances of the same type share the same configs, unless otherwise stated.

If your GPU run out of memory on large instances, use the `--micro_batch` trick, which accumulates a few batches before updating the model params. You can set e.g. `--micro_batch 2 --batchsize 4` which is in theory equivalent to `--batchsize 8`

## Bipartite ablation

We provide bipartite graphs as ablation to our tripartite approach. Which can also be considered as the implementation of the baseline from this [paper](https://openreview.net/forum?id=cP2QVK-uygd). See `run/bipartite.sh`.

## ODE baseline

We compare our MPNN approach with the neural-ODE-inspired [method](https://www.sciencedirect.com/science/article/abs/pii/S0925231222014412). See `run/ode.sh`. 
