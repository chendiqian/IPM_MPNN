import argparse
from functools import partial
from ml_collections import ConfigDict

from torch_scatter import scatter
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader as PT_DataLoader
from torch_geometric.loader import DataLoader as PyG_DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm
import wandb

from data.data_preprocess import HeteroAddLaplacianEigenvectorPE, SubSample, LogNormalize
from data.dataset import SetCoverDataset, collate_fn_ip
from data.utils import log_denormalize, args_set_bool, barrier_function
from models.hetero_gnn import TripartiteHeteroGNN, BipartiteHeteroGNN


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
    parser.add_argument('--loss', type=str, default='primal', choices=['unsupervised', 'primal', 'primal+objgap'])
    parser.add_argument('--losstype', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--use_norm', type=str, default='true')
    parser.add_argument('--use_res', type=str, default='false')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--wandbname', type=str, default='default')
    parser.add_argument('--use_wandb', type=str, default='false')
    parser.add_argument('--normalize_dataset', type=str, default='false')
    return parser.parse_args()


class Trainer:
    def __init__(self, device, loss_target, loss_type, mean, std, ipm_steps, ipm_alpha):
        assert 0. <= ipm_alpha <= 1.
        self.step_weight = torch.tensor([ipm_alpha ** (ipm_steps - l - 1)
                                         for l in range(ipm_steps)],
                                        dtype=torch.float, device=device)[None]
        self.best_val_loss = 1.e8
        self.best_val_objgap = 100.
        self.patience = 0
        self.device = device
        if loss_target != 'unsupervised':
            self.loss_target = loss_target.split('+')
        else:
            self.loss_target = loss_target
        if loss_type == 'l2':
            self.loss_func = partial(torch.pow, exponent=2)
        elif loss_type == 'l1':
            self.loss_func = torch.abs
        else:
            raise ValueError
        self.mean = mean
        self.std = std

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            optimizer.zero_grad()
            vals, cons = model(data)
            loss = self.get_loss(vals, data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=1.0,
                                           error_if_nonfinite=True)
            optimizer.step()
            train_losses += loss * data.num_graphs
            num_graphs += data.num_graphs
        return train_losses.item() / num_graphs


    @torch.no_grad()
    def eval(self, dataloader, model, scheduler):
        model.eval()

        val_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, cons = model(data)
            loss = self.get_loss(vals, data)
            val_losses += loss * data.num_graphs
            num_graphs += data.num_graphs
        val_loss = val_losses.item() / num_graphs
        scheduler.step(val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience = 0
        else:
            self.patience += 1
        return val_loss

    def get_loss(self, vals, data):
        loss = 0.
        if self.loss_target == 'unsupervised':
            # log barrier functions
            pred = vals[:, -1:]
            Ax = scatter(pred.squeeze()[data.A_col] * data.A_val, data.A_row,
                         reduce='sum', dim=0)
            loss = loss + barrier_function(data.rhs - Ax).mean()  # b - x >= 0.
            loss = loss + barrier_function(pred.squeeze()).mean()  # x >= 0.
            pred = pred * self.std + self.mean
            pred = log_denormalize(pred)
            c_times_x = data.obj_const[:, None] * pred
            obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum').mean()
            loss = loss + obj_pred
        else:
            if 'primal' in self.loss_target:
                primal_loss = (self.loss_func(vals - data.gt_primals) * self.step_weight).mean()
                loss = loss + primal_loss
            if 'objgap' in self.loss_target:
                obj_loss = (self.loss_func(self.get_obj_metric(data, vals)) * self.step_weight).mean()
                loss = loss + obj_loss
        return loss

    def get_obj_metric(self, data, pred):
        pred = pred * self.std + self.mean
        pred = log_denormalize(pred)
        c_times_x = data.obj_const[:, None] * pred
        obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')
        x_gt = data.gt_primals * self.std + self.mean
        x_gt = log_denormalize(x_gt)
        c_times_xgt = data.obj_const[:, None] * x_gt
        obj_gt = scatter(c_times_xgt, data['vals'].batch, dim=0, reduce='sum')
        return (obj_pred - obj_gt) / obj_gt

    def obj_metric(self, dataloader, model):
        model.eval()

        obj_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)
            obj_gap.append(np.abs(self.get_obj_metric(data, vals).cpu().numpy()))

        return np.concatenate(obj_gap, axis=0)


if __name__ == '__main__':
    args = args_parser()
    args = args_set_bool(vars(args))
    args = ConfigDict(args)

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

    if args.loss == 'unsupervised':
        Loader = PT_DataLoader
        collate_fn = collate_fn_ip
    else:
        Loader = PyG_DataLoader
        collate_fn = None

    train_loader = Loader(dataset[:int(len(dataset) * 0.8)],
                          batch_size=args.batchsize,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True,
                          collate_fn=collate_fn)
    val_loader = Loader(dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                        batch_size=args.batchsize,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                        collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_val_losses = []
    best_val_objgap_mean = []

    for run in range(args.runs):
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
                                        share_weight=False,
                                        use_norm=args.use_norm,
                                        use_res=args.use_res).to(device)
        
        wandb.watch(model, log="all", log_freq=10)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1.e-5)

        trainer = Trainer(device, args.loss, args.losstype, dataset.mean, dataset.std, args.ipm_steps, args.ipm_alpha)

        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            train_loss = trainer.train(train_loader, model, optimizer)
            val_loss = trainer.eval(val_loader, model, scheduler)

            if epoch % 10 == 1:
                with torch.no_grad():
                    train_gaps = trainer.obj_metric(train_loader, model)
                    val_gaps = trainer.obj_metric(val_loader, model)
                    trainer.best_val_objgap = min(trainer.best_val_objgap, val_gaps[:, -1].mean().item())
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

    torch.save(model.state_dict(), 'best_model.pt')
    wandb.log({'best_val_loss': np.mean(best_val_losses),
               'best_val_objgap': np.mean(best_val_objgap_mean)})
