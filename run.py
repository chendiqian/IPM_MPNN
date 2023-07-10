import os
import argparse
from ml_collections import ConfigDict
import yaml

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm
import wandb

from data.data_preprocess import HeteroAddLaplacianEigenvectorPE, SubSample, LogNormalize
from data.dataset import SetCoverDataset, collate_fn_ip
from data.utils import args_set_bool
from models.hetero_gnn import TripartiteHeteroGNN, BipartiteHeteroGNN
from trainer import Trainer


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--lappe', type=int, default=5)
    parser.add_argument('--ipm_restarts', type=int, default=10)
    parser.add_argument('--ipm_steps', type=int, default=8)
    parser.add_argument('--ipm_alpha', type=float, default=0.9)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1.e-3)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--use_bipartite', type=str, default='false')
    parser.add_argument('--loss', type=str, default='primal+objgap',
                        choices=['unsupervised', 'primal', 'primal+objgap', 'primal+objgap+constraint'])
    parser.add_argument('--loss_weight_x', type=float, default=1.0)
    parser.add_argument('--loss_weight_obj', type=float, default=1.0)
    parser.add_argument('--loss_weight_cons', type=float, default=0.0)  # does not work
    parser.add_argument('--losstype', type=str, default='l2', choices=['l1', 'l2'])  # no big different
    parser.add_argument('--dropout', type=float, default=0.)  # must
    parser.add_argument('--use_norm', type=str, default='true')  # must
    parser.add_argument('--use_res', type=str, default='false')  # does not help
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--wandbname', type=str, default='default')
    parser.add_argument('--use_wandb', type=str, default='false')
    parser.add_argument('--normalize_dataset', type=str, default='false')  # does not help

    parser.add_argument('--share_lin_weight', type=str, default='false')
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    args = args_set_bool(vars(args))
    args = ConfigDict(args)

    if not os.path.isdir('logs'):
        os.mkdir('logs')
    exist_runs = [d for d in os.listdir('logs') if d.startswith('exp')]
    log_folder_name = f'logs/exp{len(exist_runs)}'
    os.mkdir(log_folder_name)
    with open(os.path.join(log_folder_name, 'config.yaml'), 'w') as outfile:
        yaml.dump(args.to_dict(), outfile, default_flow_style=False)

    wandb.init(project=args.wandbname, mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="ipmgnn")

    dataset = SetCoverDataset(args.datapath,
                              normalize=args.normalize_dataset,
                              rand_starts=args.ipm_restarts,
                              transform=SubSample(args.ipm_steps),
                              pre_transform=Compose([HeteroAddLaplacianEigenvectorPE(k=args.lappe),
                                                     SubSample(8),
                                                     LogNormalize()]))

    train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)],
                              batch_size=args.batchsize,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              collate_fn=collate_fn_ip)
    val_loader = DataLoader(dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                            batch_size=args.batchsize,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=collate_fn_ip)
    test_loader = DataLoader(dataset[int(len(dataset) * 0.9):],
                            batch_size=args.batchsize,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=collate_fn_ip)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_val_losses = []
    best_val_objgap_mean = []
    test_losses = []
    test_objgap_mean = []

    for run in range(args.runs):
        os.mkdir(os.path.join(log_folder_name, f'run{run}'))
        if args.use_bipartite:
            model = BipartiteHeteroGNN(in_shape=2,
                                       pe_dim=args.lappe,
                                       hid_dim=args.hidden,
                                       num_layers=args.ipm_steps,
                                       use_norm=args.use_norm).to(device)
        else:
            model = TripartiteHeteroGNN(in_shape=2,
                                        pe_dim=args.lappe,
                                        hid_dim=args.hidden,
                                        num_layers=args.ipm_steps,
                                        dropout=args.dropout,
                                        share_conv_weight=False,
                                        share_lin_weight=args.share_lin_weight,
                                        use_norm=args.use_norm,
                                        use_res=args.use_res).to(device)
        
        # wandb.watch(model, log="all", log_freq=10)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1.e-5)

        trainer = Trainer(device,
                          args.loss,
                          args.losstype,
                          dataset.mean, dataset.std,
                          args.ipm_steps,
                          args.ipm_alpha,
                          loss_weight={'primal': args.loss_weight_x,
                                       'objgap': args.loss_weight_obj,
                                       'constraint': args.loss_weight_cons})

        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            train_loss = trainer.train(train_loader, model, optimizer)
            val_loss = trainer.eval(val_loader, model, scheduler)

            if epoch % 10 == 1:
                with torch.no_grad():
                    train_gaps = trainer.obj_metric(train_loader, model)
                    val_gaps = trainer.obj_metric(val_loader, model)
                    cur_mean_gap = val_gaps[:, -1].mean().item()
                    if trainer.best_val_objgap > cur_mean_gap:
                        trainer.best_val_objgap = cur_mean_gap
                        torch.save(model.state_dict(), os.path.join(log_folder_name, f'run{run}', 'best_model.pt'))
            else:
                train_gaps, val_gaps = None, None

            if trainer.patience > args.patience:
                break

            pbar.set_postfix({'train_loss': train_loss, 'val_loss': val_loss, 'lr': scheduler.optimizer.param_groups[0]["lr"]})
            log_dict = {'train_loss': train_loss,
                       'val_loss': val_loss,
                       'lr': scheduler.optimizer.param_groups[0]["lr"]}
            if train_gaps is not None:
                for gnn_l in range(train_gaps.shape[1]):
                    log_dict[f'train_obj_gap_l{gnn_l}_mean'] = train_gaps[:, gnn_l].mean()
                    log_dict[f'train_obj_gap_l{gnn_l}'] = wandb.Histogram(train_gaps[:, gnn_l])
            if val_gaps is not None:
                for gnn_l in range(val_gaps.shape[1]):
                    log_dict[f'val_obj_gap_l{gnn_l}_mean'] = val_gaps[:, gnn_l].mean()
                    log_dict[f'val_obj_gap_l{gnn_l}'] = wandb.Histogram(val_gaps[:, gnn_l])
            wandb.log(log_dict)
        best_val_losses.append(trainer.best_val_loss)
        best_val_objgap_mean.append(trainer.best_val_objgap)

        model.load_state_dict(torch.load(os.path.join(log_folder_name, f'run{run}', 'best_model.pt'), map_location=device))
        with torch.no_grad():
            test_loss = trainer.eval(test_loader, model, scheduler, test=True)
            test_gaps = trainer.obj_metric(test_loader, model)
        test_losses.append(test_loss)
        test_objgap_mean.append(test_gaps[:, -1].mean().item())

    wandb.log({'best_val_loss': np.mean(best_val_losses),
               'best_val_objgap': np.mean(best_val_objgap_mean),
               'test_loss_mean': np.mean(test_losses),
               'test_loss_std': np.std(test_losses),
               'test_objgap_mean': np.mean(test_objgap_mean),
               'test_objgap_std': np.std(test_objgap_mean)})
