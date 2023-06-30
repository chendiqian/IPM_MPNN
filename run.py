import argparse

from torch_scatter import scatter
import numpy as np
import torch
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm
import wandb

from data.data_preprocess import HeteroAddLaplacianEigenvectorPE, SubSample, LogNormalize
from data.dataset import SetCoverDataset
from data.utils import log_denormalize, mode_of_distribution
from models.parallel_hetero_gnn import ParallelHeteroGNN
from models.async_bipartite_gnn import UnParallelHeteroGNN


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--lappe', type=int, default=5)
    parser.add_argument('--ipm_steps', type=int, default=8)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1.e-3)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--use_bipartite', type=bool, default=False)
    parser.add_argument('--parallel', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--use_norm', type=bool, default=False)
    parser.add_argument('--patience', type=int, default=100)
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
            loss = self.criterion(vals, data.gt_primals)
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
            loss = self.criterion(vals, data.gt_primals)
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

    @torch.no_grad()
    def obj_metric(self, dataloader, model):
        model.eval()

        obj_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)
            vals = log_denormalize(vals)
            c_times_x = data.obj_const[:, None] * vals
            obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')
            x_gt = log_denormalize(data.gt_primals)
            c_times_xgt = data.obj_const[:, None] * x_gt
            obj_gt = scatter(c_times_xgt, data['vals'].batch, dim=0, reduce='sum')
            assert torch.allclose(obj_gt[:, -1], data.obj_value, rtol=1.e-3, atol=1.e-5)
            obj_gap.append(np.abs(((obj_gt - obj_pred) / obj_gt).cpu().numpy()))

        return np.concatenate(obj_gap, axis=0)


if __name__ == '__main__':
    args = args_parser()
    wandb.init(project=args.wandbname, mode="online" if args.use_wandb else "disabled",
               config=vars(args),
               entity="ipmgnn")

    dataset = SetCoverDataset(args.datapath,
                              transform=SubSample(args.ipm_steps),
                              pre_transform=Compose([HeteroAddLaplacianEigenvectorPE(k=args.lappe),
                                                     SubSample(8),
                                                     LogNormalize()]))

    train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)], batch_size=args.batchsize, shuffle=True)
    val_loader = DataLoader(dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)], batch_size=args.batchsize, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_val_losses = []

    for run in range(args.runs):
        if not args.parallel:
            model = UnParallelHeteroGNN(in_shape=2,
                                        pe_dim=args.lappe,
                                        hid_dim=args.hidden,
                                        num_layers=args.ipm_steps,
                                        use_norm=args.use_norm).to(device)
        else:
            model = ParallelHeteroGNN(bipartite=args.use_bipartite,
                                      in_shape=2,
                                      pe_dim=args.lappe,
                                      hid_dim=args.hidden,
                                      num_layers=args.ipm_steps,
                                      dropout=args.dropout,
                                      share_weight=False,
                                      use_norm=args.use_norm,
                                      use_res=False).to(device)
        
        wandb.watch(model, log="all", log_freq=10)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1.e-5)

        trainer = Trainer(device, torch.nn.MSELoss())

        pbar = tqdm(range(args.epoch))
        for epoch in pbar:
            train_loss = trainer.train(train_loader, model, optimizer)
            val_loss = trainer.eval(val_loader, model, scheduler)

            if epoch % 10 == 1:
                train_gaps = trainer.obj_metric(train_loader, model)
                val_gaps = trainer.obj_metric(val_loader, model)
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
                    log_dict[f'train_obj_gap_l{gnn_l}_mode'] = mode_of_distribution(train_gaps[:, gnn_l])
            if val_gaps is not None:
                for gnn_l in range(val_gaps.shape[1]):
                    log_dict[f'val_obj_gap_l{gnn_l}_mean'] = val_gaps[:, gnn_l].mean()
                    log_dict[f'val_obj_gap_l{gnn_l}_mode'] = mode_of_distribution(val_gaps[:, gnn_l])
            wandb.log(log_dict)
        best_val_losses.append(trainer.best_val_loss)

    print(f'best loss: {np.mean(best_val_losses)} ± {np.std(best_val_losses)}')
