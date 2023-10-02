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

# Reproduction of the empirical results

## setcover
`python run.py --datapath DATA_TO_YOUR_INSTANCES --upper 1. --ipm_alpha 0.15 --weight_decay 1.2e-6 --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1.2 --loss_weight_obj 0.8 --loss_weight_cons 0.16 --runs 3 --lappe 0`

## indset
`python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.15 --weight_decay 1.2e-6 --upper 1. --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1.2 --loss_weight_obj 0.8 --loss_weight_cons 0.16 --runs 3 --lappe 0`

## cauction
`python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.8 --weight_decay 1.6e-8 --upper 1. --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 4 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.17 --loss_weight_cons 4.72 --runs 3 --lappe 0`

## fac
`python run.py --datapath DATA_TO_YOUR_INSTANCES --ipm_alpha 0.8 --weight_decay 3.8e-6 --upper 1. --batchsize 512 --hidden 180 --num_pred_layers 4 --num_mlp_layers 2 --share_lin_weight false --conv_sequence cov --loss_weight_x 1. --loss_weight_obj 2.99 --loss_weight_cons 8.15 --runs 3 --lappe 0`

Note that the small and large instances of the same type share the same configs. 

If your GPU run out of memory on large instances, use the `--micro_batch` trick, which accumulates a few batches before updating the model params. You can set e.g. `--micro_batch 2 --batchsize 4` which is in theory equivalent to `--batchsize 8`
