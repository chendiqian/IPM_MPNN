#!/bin/bash

### GENConv

### setcover
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0. --weight_decay 1.2e-6 --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1.2 --loss_weight_obj 0.8 --loss_weight_cons 0.16 --runs 3 --bipartite true

### indset
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0. --weight_decay 1.2e-6 --upper 1. --use_wandb true --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1.2 --loss_weight_obj 0.8 --loss_weight_cons 0.16 --runs 3 --bipartite true

### cauction
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0. --weight_decay 1.6e-8 --upper 1. --use_wandb true --batchsize 256 --micro_batch 2 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.17 --loss_weight_cons 4.72 --runs 3 --bipartite true

### fac
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0. --weight_decay 3.8e-6 --upper 1. --use_wandb true --batchsize 256 --micro_batch 2 --hidden 180 --num_pred_layers 4 --num_mlp_layers 2 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.99 --loss_weight_cons 8.15 --runs 3 --bipartite true

### GCNConv

#### setcover
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.86 --weight_decay 1.e-5 --batchsize 512 --hidden 32 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --loss_weight_x 1. --loss_weight_obj 5.5 --loss_weight_cons 1.1 --runs 3 --conv gcnconv --bipartite true

#### fac
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.68 --weight_decay 0. --batchsize 512 --hidden 96 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --loss_weight_x 1. --loss_weight_obj 5.3 --loss_weight_cons 3.8 --runs 3 --conv gcnconv --bipartite true

#### indset
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.68 --weight_decay 0. --batchsize 512 --hidden 96 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --loss_weight_x 1. --loss_weight_obj 6.3 --loss_weight_cons 3.1 --runs 3 --conv gcnconv --bipartite true

#### cauc
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.65 --weight_decay 1.e-7 --batchsize 512 --hidden 128 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --loss_weight_x 1. --loss_weight_obj 4.7 --loss_weight_cons 4.3 --runs 3 --conv gcnconv --bipartite true

### GINConv

#### setcover
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.3 --weight_decay 1.e-5 --batchsize 512 --hidden 64 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --loss_weight_x 1. --loss_weight_obj 4.7 --loss_weight_cons 0.8 --runs 3 --conv ginconv --bipartite true

#### indset
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.73 --weight_decay 5.6e-6 --batchsize 512 --hidden 180 --num_pred_layers 3 --num_mlp_layers 2 --share_lin_weight false --loss_weight_x 1. --loss_weight_obj 2.4 --loss_weight_cons 7.5 --runs 3 --conv ginconv --bipartite true

#### cauc
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.43 --weight_decay 1.2e-8 --batchsize 512 --hidden 128 --num_pred_layers 3 --num_mlp_layers 4 --share_lin_weight false --loss_weight_x 1. --loss_weight_obj 6.2 --loss_weight_cons 4.1 --runs 3 --conv ginconv --bipartite true

#### fac
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.8 --weight_decay 1.e-7 --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --loss_weight_x 1. --loss_weight_obj 1.3 --loss_weight_cons 4.6 --runs 3 --conv ginconv --bipartite true