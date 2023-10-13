import argparse
import os
from functools import partial

import copy
import numpy as np
import torch
import wandb
import yaml
from functorch.experimental import replace_all_batch_norm_modules_
from ml_collections import ConfigDict
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
from torch_sparse import SparseTensor
from tqdm import tqdm

from trainer import Trainer
from data.data_preprocess import HeteroAddLaplacianEigenvectorPE, SubSample
from data.dataset import LPDataset
from data.utils import args_set_bool, collate_fn_with_counts
from models.time_depend_gnn import TimeDependentTripartiteHeteroGNN


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    # admin
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--wandbproject', type=str, default='default')
    parser.add_argument('--wandbname', type=str, default='')
    parser.add_argument('--use_wandb', type=str, default='false')

    # ipm processing
    parser.add_argument('--ipm_restarts', type=int, default=1)  # more does not help
    parser.add_argument('--ipm_steps', type=int, default=3)
    parser.add_argument('--upper', type=float, default=1.0)

    # training dynamics
    parser.add_argument('--ckpt', type=str, default='true')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1.e-3)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--micro_batch', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.)  # must
    parser.add_argument('--use_norm', type=str, default='true')  # must
    parser.add_argument('--use_res', type=str, default='false')  # does not help
    parser.add_argument('--T', type=float, default=100.)
    parser.add_argument('--repeats', type=int, default=8)  # repeat of t sampled

    # model related
    parser.add_argument('--conv', type=str, default='genconv')
    parser.add_argument('--lappe', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=3)
    parser.add_argument('--num_pred_layers', type=int, default=2)
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='mlp layers within GENConv')
    parser.add_argument('--share_conv_weight', type=str, default='false')
    parser.add_argument('--conv_sequence', type=str, default='cov')

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    args = args_set_bool(vars(args))
    args = ConfigDict(args)

    if args.ckpt:
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        exist_runs = [d for d in os.listdir('logs') if d.startswith('exp')]
        log_folder_name = f'logs/exp{len(exist_runs)}'
        os.mkdir(log_folder_name)
        with open(os.path.join(log_folder_name, 'config.yaml'), 'w') as outfile:
            yaml.dump(args.to_dict(), outfile, default_flow_style=False)

    wandb.init(project=args.wandbproject,
               name=args.wandbname if args.wandbname else None,
               mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="chendiqian")  # use your own entity

    dataset = LPDataset(args.datapath,
                        extra_path=f'{args.ipm_restarts}restarts_'
                                         f'{args.lappe}lap_'
                                         f'{args.ipm_steps}steps'
                                         f'{"_upper_" + str(args.upper) if args.upper is not None else ""}',
                        upper_bound=args.upper,
                        rand_starts=args.ipm_restarts,
                        pre_transform=Compose([HeteroAddLaplacianEigenvectorPE(k=args.lappe),
                                                     SubSample(args.ipm_steps)]))

    train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)],
                              batch_size=args.batchsize,
                              shuffle=True,
                              num_workers=1,
                              collate_fn=collate_fn_with_counts)
    val_loader = DataLoader(dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                            batch_size=args.batchsize,
                            shuffle=False,
                            num_workers=1,
                            collate_fn=collate_fn_with_counts)
    test_loader = DataLoader(dataset[int(len(dataset) * 0.9):],
                             batch_size=args.batchsize,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=collate_fn_with_counts)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_val_objgap_mean = []
    best_val_consgap_mean = []
    test_objgap_mean = []
    test_consgap_mean = []

    for run in range(args.runs):
        if args.ckpt:
            os.mkdir(os.path.join(log_folder_name, f'run{run}'))
        model = TimeDependentTripartiteHeteroGNN(conv=args.conv,
                                                 in_shape=2,
                                                 pe_dim=args.lappe,
                                                 hid_dim=args.hidden,
                                                 num_conv_layers=args.num_conv_layers,
                                                 num_pred_layers=args.num_pred_layers,
                                                 num_mlp_layers=args.num_mlp_layers,
                                                 dropout=args.dropout,
                                                 share_conv_weight=args.share_conv_weight,
                                                 use_norm=args.use_norm,
                                                 use_res=args.use_res,
                                                 conv_sequence=args.conv_sequence).to(device)
        model = replace_all_batch_norm_modules_(model)
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1.e-5)

        trainer = Trainer(device,
                          '',
                          'l2',
                          args.micro_batch,
                          min(args.ipm_steps, args.num_conv_layers),
                          1.,
                          loss_weight=None)

        pbar = tqdm(range(args.epochs))
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()

            update_count = 0
            micro_batch = int(min(args.micro_batch, len(train_loader)))
            loss_scaling_lst = [micro_batch] * (len(train_loader) // micro_batch) + [len(train_loader) % micro_batch]

            train_losses = 0.
            num_graphs = 0
            for i, data in enumerate(train_loader):
                data = data.to(device)
                time_var = torch.rand(args.repeats).to(device) * args.T

                forward = partial(model.forward, data=data)
                jac = torch.func.jacrev(forward, argnums=0)
                batch_jac = torch.vmap(jac, in_dims=0, out_dims=1)
                jac = batch_jac(time_var)

                def split(t):
                    batch_forward = torch.vmap(forward, in_dims=0, out_dims=1)
                    val_con_repeats = batch_forward(t)
                    vals, cons = torch.split(val_con_repeats,
                                             torch.hstack([data.num_val_nodes.sum(),
                                                           data.num_con_nodes.sum()]).tolist(), dim=0)
                    return vals, cons

                x, u = split(time_var)
                A = SparseTensor(row=data.A_row, col=data.A_col, value=data.A_val)
                D = data.obj_const
                b = data.rhs
                uaxb = torch.relu(u + A @ x - b[:, None])
                phi = torch.cat([-(D[:, None] + A.t() @ uaxb), uaxb - u], dim=0)

                loss = torch.nn.MSELoss()(jac, phi)

                train_losses += loss.detach() * data.num_graphs
                num_graphs += data.num_graphs

                update_count += 1
                loss = loss / float(loss_scaling_lst[0])  # scale the loss
                loss.backward()

                if update_count >= micro_batch or i == len(train_loader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   max_norm=1.0,
                                                   error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad()
                    update_count = 0
                    loss_scaling_lst.pop(0)

            train_loss = train_losses.item() / num_graphs

            val_obj_gaps, val_constraint_gaps = trainer.eval_baseline(val_loader, model, args.T)

            cur_mean_gap = val_obj_gaps.mean().item()
            cur_cons_mean_gap = val_constraint_gaps.mean().item()
            if scheduler is not None:
                scheduler.step(cur_mean_gap)

            if trainer.best_val_objgap > cur_mean_gap:
                trainer.patience = 0
                trainer.best_val_objgap = cur_mean_gap
                trainer.best_val_consgap = cur_cons_mean_gap
                best_model = copy.deepcopy(model.state_dict())
                if args.ckpt:
                    torch.save(model.state_dict(), os.path.join(log_folder_name, f'run{run}', 'best_model.pt'))
            else:
                trainer.patience += 1

            if trainer.patience > args.patience:
                break

            pbar.set_postfix({'train_loss': train_loss,
                              'val_obj_gap': cur_mean_gap,
                              'val_cons_gap': cur_cons_mean_gap,
                              'lr': scheduler.optimizer.param_groups[0]["lr"]})
            log_dict = {'train_loss': train_loss,
                        'val_obj_gap': cur_mean_gap,
                        'val_cons_gap': cur_cons_mean_gap,
                        'lr': scheduler.optimizer.param_groups[0]["lr"]}
            wandb.log(log_dict)
        best_val_objgap_mean.append(trainer.best_val_objgap)
        best_val_consgap_mean.append(trainer.best_val_consgap)

        model.load_state_dict(best_model)

        test_obj_gaps, test_constraint_gaps = trainer.eval_baseline(test_loader, model, args.T)
        test_objgap_mean.append(test_obj_gaps.mean().item())
        test_consgap_mean.append(test_constraint_gaps.mean().item())

        wandb.log({'test_objgap': test_objgap_mean[-1]})
        wandb.log({'test_consgap': test_consgap_mean[-1]})

    wandb.log({
        'test_objgap_mean': np.mean(test_objgap_mean),
        'test_objgap_std': np.std(test_objgap_mean),
        'test_consgap_mean': np.mean(test_consgap_mean),
        'test_consgap_std': np.std(test_consgap_mean),
    })
