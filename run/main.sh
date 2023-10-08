#!/bin/bash

# GEN

#### setcover
python run.py --datapath DATA_TO_YOUR_INSTANCES --upper 1. --ipm_alpha 0.15 --weight_decay 1.2e-6 --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1.2 --loss_weight_obj 0.8 --loss_weight_cons 0.16 --runs 3 --conv genconv

#### indset
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.15 --weight_decay 1.2e-6 --upper 1. --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1.2 --loss_weight_obj 0.8 --loss_weight_cons 0.16 --runs 3 --conv genconv

#### cauction
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.86 --weight_decay 0. --batchsize 512  --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --loss_weight_x 1. --loss_weight_obj 4.6 --loss_weight_cons 5.3 --runs 3 --conv genconv

#### fac
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.8 --weight_decay 3.8e-6 --upper 1. --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 2 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.99 --loss_weight_cons 8.15 --runs 3 --conv genconv

# GCNConv

#### small setcover
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.76 --weight_decay 4.4e-7 --batchsize 512 --hidden 180 --num_pred_layers 3 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 0.33 --loss_weight_cons 2.2 --runs 3 --lappe 0 --conv gcnconv

#### large setcover
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.2 --weight_decay 1.5e-8 --batchsize 128 --micro_batch 4 --hidden 180 --num_pred_layers 3 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.2 --loss_weight_cons 0.26 --runs 3 --conv gcnconv

#### indset
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.5 --weight_decay 2.e-7 --upper 1. --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 4.5 --loss_weight_cons 9.6 --runs 3 --conv gcnconv

#### cauction
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.35 --weight_decay 0. --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --loss_weight_x 1. --loss_weight_obj 3.43 --loss_weight_cons 5.8 --runs 3 --conv gcnconv

#### fac
`python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.63 --weight_decay 4.5e-7 --upper 1. --batchsize 512 --hidden 96 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 8.68 --loss_weight_cons 9.56 --runs 3 --conv gcnconv`

# GINconv

#### small setcover
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.73 --weight_decay 5.6e-6 --batchsize 512 --hidden 180 --num_pred_layers 3 --num_mlp_layers 2 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.4 --loss_weight_cons 7.5 --runs 3 --conv ginconv

#### large setcover
python run.py --datapath DATA_TO_YOUR_INSTANCES  --ipm_alpha 0.7 --weight_decay 2.8e-8 --batchsize 80 --micro_batch 4 --hidden 180 --num_pred_layers 3 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 4.5 --loss_weight_cons 2.2 --runs 3 --conv ginconv

#### indset
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.73 --weight_decay 5.6e-6 --batchsize 512 --hidden 180 --num_pred_layers 3 --num_mlp_layers 2 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.4 --loss_weight_cons 7.5 --runs 3 --conv ginconv

#### cauction
python run.py --datapath DATA_TO_YOUR_INSTANCES  --ipm_alpha 0.63 --weight_decay 0. --batchsize 512  --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --loss_weight_x 1. --loss_weight_obj 4.3 --loss_weight_cons 6.26 --runs 3 --conv ginconv

#### small fac
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.8 --weight_decay 1.e-7 --batchsize 400 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 1.3 --loss_weight_cons 4.6 --runs 3 --conv ginconv

#### large fac
python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.9 --weight_decay 1.e-5 --batchsize 160 --micro_batch 3 --hidden 128 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.5 --loss_weight_cons 4.0 --runs 3 --conv ginconv