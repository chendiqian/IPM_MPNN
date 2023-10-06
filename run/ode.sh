#!/bin/bash

# GENConv

### setcover
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.28 --weight_decay 3.36e-3 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 128 --num_pred_layers 2 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 3.5 --loss_weight_cons 1.3 --runs 3

### indset
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.38 --weight_decay 1.e-6 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 128 --num_pred_layers 2 --num_mlp_layers 2 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 7.1 --loss_weight_cons 6.2 --runs 3

### cauc
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.85 --weight_decay 8.e-5 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 128 --num_pred_layers 4 --num_mlp_layers 2 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 9.6 --loss_weight_cons 7.1 --runs 3

### fac
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.67 --weight_decay 1.7e-6 --ipm_steps 3 --num_conv_layers 3 --batchsize 512 --hidden 128 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 5.3 --loss_weight_cons 0.75 --runs 3

### the baselines
python baseline.py --datapath DATA_TO_YOUR_INSTANCES --ipm_steps 3 --batchsize 8 --hidden 128 --num_pred_layers 2 --num_mlp_layers 2 --repeats 4 --num_conv_layers 3 --runs 3
