#!/bin/bash

# GENConv

### setcover
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.28 --weight_decay 3.36e-3 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 128 --num_pred_layers 2 --num_mlp_layers 4 --loss_weight_x 1. --loss_weight_obj 3.5 --loss_weight_cons 1.3 --runs 3

### indset
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.38 --weight_decay 1.e-6 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 128 --num_pred_layers 2 --num_mlp_layers 2 --loss_weight_x 1. --loss_weight_obj 7.1 --loss_weight_cons 6.2 --runs 3

### cauc
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.85 --weight_decay 8.e-5 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 128 --num_pred_layers 4 --num_mlp_layers 2 --loss_weight_x 1. --loss_weight_obj 9.6 --loss_weight_cons 7.1 --runs 3

### fac
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.67 --weight_decay 1.7e-6 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 128 --num_pred_layers 4 --num_mlp_layers 4 --loss_weight_x 1. --loss_weight_obj 5.3 --loss_weight_cons 0.75 --runs 3

### the baselines
python baseline.py --datapath DATA_TO_YOUR_INSTANCES --ipm_steps 3 --batchsize 8 --hidden 128 --num_pred_layers 2 --num_mlp_layers 2 --repeats 4 --num_conv_layers 3 --runs 3

# GCNCov

### setcover
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.78 --weight_decay 1.e-6 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 180 --num_pred_layers 3 --num_mlp_layers 3 --loss_weight_x 1. --loss_weight_obj 6.07 --loss_weight_cons 1.60 --runs 3 --conv gcnconv

### indset
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.78 --weight_decay 2.1e-6 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 180 --num_pred_layers 2 --num_mlp_layers 4 --loss_weight_x 1. --loss_weight_obj 3.53 --loss_weight_cons 5.61 --runs 3 --conv gcnconv

### cauc
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.65 --weight_decay 3.7e-6 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 3 --loss_weight_x 1. --loss_weight_obj 4.66 --loss_weight_cons 5.01 --runs 3 --conv gcnconv

### fac
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.83 --weight_decay 9.2e-6 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 128 --num_pred_layers 4 --num_mlp_layers 2 --loss_weight_x 1. --loss_weight_obj 2.98 --loss_weight_cons 3.71 --runs 3 --conv gcnconv

# GINConv
### setcover
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.36 --weight_decay 5.5e-6 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 180 --num_pred_layers 2 --num_mlp_layers 3 --loss_weight_x 1. --loss_weight_obj 3.77 --loss_weight_cons 1.39 --runs 3 --conv ginconv

### indset
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.44 --weight_decay 9.7e-6 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 3 --loss_weight_x 1. --loss_weight_obj 5.88 --loss_weight_cons 3.88 --runs 3 --conv ginconv

### cauc
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.89 --weight_decay 1.e-6 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 128 --num_pred_layers 3 --num_mlp_layers 2 --loss_weight_x 1. --loss_weight_obj 6.42 --loss_weight_cons 5.04 --runs 3 --conv ginconv

### fac
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.82 --weight_decay 3.56e-7 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 3 --loss_weight_x 1. --loss_weight_obj 1.77 --loss_weight_cons 1.45 --runs 3 --conv ginconv