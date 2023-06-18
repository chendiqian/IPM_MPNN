import argparse

import numpy as np
import torch
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm
import wandb

from data.data_preprocess import HeteroAddLaplacianEigenvectorPE, SubSample
from data.dataset import SetCoverDataset
from models import DeepHeteroGNN


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    parser.add_argument('--lappe', type=int, default=5)
    parser.add_argument('--ipm_steps', type=int, default=8)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1.e-3)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--wandbname', type=str, default='default')
    parser.add_argument('--use_wandb', type=str, default=False)
    return parser.parse_args()


class Trainer:
    def __init__(self, device, criterion):
        self.best_val_loss = 1.e8
        self.patience = 0
        self.device = device
        self.criterion = criterion

    def train(self, dataloader, model, optimizer):
        model.train()

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            optimizer.zero_grad()
            vals, cons = model(data)
            loss = self.criterion(vals[..., 0], data.gt_primals)
            loss.backward()
            optimizer.step()
            train_losses += loss * data.num_graphs
            num_graphs += data.num_graphs
        return train_losses.item() / num_graphs


    @torch.no_grad()
    def eval(self, dataloader, model, scheduler):
        val_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, cons = model(data)
            loss = self.criterion(vals[..., 0], data.gt_primals)
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


if __name__ == '__main__':
    args = args_parser()
    wandb.init(project=args.wandbname, mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="ipmgnn")

    dataset = SetCoverDataset('instances/setcover',
                              transform=SubSample(args.ipm_steps),
                              pre_transform=Compose([HeteroAddLaplacianEigenvectorPE(k=args.lappe),
                                                     SubSample(32)]))

    train_loader = DataLoader(dataset[:800], batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(dataset[800:900], batch_size=args.batchsize, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_val_losses = []

    for run in range(args.runs):
        model = DeepHeteroGNN(in_shape=2,
                              pe_dim=args.lappe,
                              hid_dim=args.hidden,
                              num_layers=args.ipm_steps,
                              dropout=args.dropout,
                              share_weight=False,
                              use_norm=False,
                              use_res=False).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1.e-5)

        trainer = Trainer(device, torch.nn.MSELoss())

        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            train_loss = trainer.train(train_loader, model, optimizer)
            val_loss = trainer.eval(val_loader, model, scheduler)
            if trainer.patience > 100:
                break

            pbar.set_postfix({'train loss': train_loss, 'val loss': val_loss, 'lr': scheduler.optimizer.param_groups[0]["lr"]})
            wandb.log({'train loss': train_loss, 'val loss': val_loss, 'lr': scheduler.optimizer.param_groups[0]["lr"]})
        best_val_losses.append(trainer.best_val_loss)

    print(f'best loss: {np.mean(best_val_losses)} ± {np.std(best_val_losses)}')
