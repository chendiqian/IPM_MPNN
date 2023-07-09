from functools import partial

import numpy as np
import torch
from torch_scatter import scatter

from data.utils import barrier_function, log_denormalize


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
            Ax = scatter(pred.squeeze()[data.A_col[data.A_tilde_mask]] *
                         data.A_val[data.A_tilde_mask],
                         data.A_row[data.A_tilde_mask],
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
                obj_loss = (self.loss_func(self.get_obj_metric(data, vals, hard_non_negative=False)) * self.step_weight).mean()
                loss = loss + obj_loss
            if 'constraint' in self.loss_target:
                pred = vals * self.std + self.mean
                pred = log_denormalize(pred)
                Ax = scatter(pred[data.A_col, :] * data.A_val[:, None], data.A_row, reduce='sum', dim=0)
                constraint_gap = data.rhs[:, None] - Ax
                cons_loss = (self.loss_func(constraint_gap) * self.step_weight).mean()
                loss = loss + cons_loss
        return loss

    def get_obj_metric(self, data, pred, hard_non_negative=False):
        # if hard_non_negative, we need a relu to make x all non-negative
        # just for metric usage, not for training
        pred = pred * self.std + self.mean
        pred = log_denormalize(pred)
        if hard_non_negative:
            pred = torch.relu(pred)
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
            obj_gap.append(np.abs(self.get_obj_metric(data, vals, hard_non_negative=True).cpu().numpy()))

        return np.concatenate(obj_gap, axis=0)
