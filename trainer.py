from functools import partial

import numpy as np
import torch
from torch_scatter import scatter

from data.utils import barrier_function


class Trainer:
    def __init__(self,
                 device,
                 loss_target,
                 loss_type,
                 micro_batch,
                 ipm_steps,
                 ipm_alpha,
                 loss_weight):
        assert 0. <= ipm_alpha <= 1.
        self.ipm_steps = ipm_steps
        self.step_weight = torch.tensor([ipm_alpha ** (ipm_steps - l - 1)
                                         for l in range(ipm_steps)],
                                        dtype=torch.float, device=device)[None]
        # self.best_val_loss = 1.e8
        self.best_val_objgap = 100.
        self.best_val_consgap = 100.
        self.patience = 0
        self.device = device
        self.loss_target = loss_target.split('+')
        self.loss_weight = loss_weight
        if loss_type == 'l2':
            self.loss_func = partial(torch.pow, exponent=2)
        elif loss_type == 'l1':
            self.loss_func = torch.abs
        else:
            raise ValueError
        self.micro_batch = micro_batch

    def train(self, dataloader, model, optimizer):
        model.train()
        optimizer.zero_grad()

        update_count = 0
        micro_batch = int(min(self.micro_batch, len(dataloader)))
        loss_scaling_lst = [micro_batch] * (len(dataloader) // micro_batch) + [len(dataloader) % micro_batch]

        train_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)
            loss = self.get_loss(vals, data)

            train_losses += loss.detach() * data.num_graphs
            num_graphs += data.num_graphs

            update_count += 1
            loss = loss / float(loss_scaling_lst[0])  # scale the loss
            loss.backward()

            if update_count >= micro_batch or i == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0,
                                               error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad()
                update_count = 0
                loss_scaling_lst.pop(0)

        return train_losses.item() / num_graphs


    @torch.no_grad()
    def eval(self, dataloader, model, scheduler = None):
        model.eval()

        val_losses = 0.
        num_graphs = 0
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)
            loss = self.get_loss(vals, data)
            val_losses += loss * data.num_graphs
            num_graphs += data.num_graphs
        val_loss = val_losses.item() / num_graphs

        if scheduler is not None:
            scheduler.step(val_loss)
        return val_loss

    def get_loss(self, vals, data):
        loss = 0.

        if 'obj' in self.loss_target:
            pred = vals[:, -self.ipm_steps:]
            c_times_x = data.obj_const[:, None] * pred
            obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')
            obj_pred = (self.loss_func(obj_pred) * self.step_weight).mean()
            loss = loss + obj_pred
        if 'barrier' in self.loss_target:
            raise NotImplementedError("Need to discuss only on the last step or on all")
            # pred = vals * self.std + self.mean
            # Ax = scatter(pred.squeeze()[data.A_col[data.A_tilde_mask]] *
            #              data.A_val[data.A_tilde_mask],
            #              data.A_row[data.A_tilde_mask],
            #              reduce='sum', dim=0)
            # loss = loss + barrier_function(data.rhs - Ax).mean()  # b - x >= 0.
            # loss = loss + barrier_function(pred.squeeze()).mean()  # x >= 0.
        if 'primal' in self.loss_target:
            primal_loss = (self.loss_func(
                vals[:, -self.ipm_steps:] -
                data.gt_primals[:, -self.ipm_steps:]
            ) * self.step_weight).mean()
            loss = loss + primal_loss * self.loss_weight['primal']
        if 'objgap' in self.loss_target:
            obj_loss = (self.loss_func(self.get_obj_metric(data, vals, hard_non_negative=False)) * self.step_weight).mean()
            loss = loss + obj_loss * self.loss_weight['objgap']
        if 'constraint' in self.loss_target:
            constraint_gap = self.get_constraint_violation(vals, data)
            cons_loss = (self.loss_func(constraint_gap) * self.step_weight).mean()
            loss = loss + cons_loss * self.loss_weight['constraint']
        return loss

    def get_constraint_violation(self, vals, data):
        """
        Ax - b

        :param vals:
        :param data:
        :return:
        """
        pred = vals[:, -self.ipm_steps:]
        Ax = scatter(pred[data.A_col, :] * data.A_val[:, None], data.A_row, reduce='sum', dim=0)
        constraint_gap = Ax - data.rhs[:, None]
        constraint_gap = torch.relu(constraint_gap)
        return constraint_gap

    def get_obj_metric(self, data, pred, hard_non_negative=False):
        # if hard_non_negative, we need a relu to make x all non-negative
        # just for metric usage, not for training
        pred = pred[:, -self.ipm_steps:]
        if hard_non_negative:
            pred = torch.relu(pred)
        c_times_x = data.obj_const[:, None] * pred
        obj_pred = scatter(c_times_x, data['vals'].batch, dim=0, reduce='sum')
        x_gt = data.gt_primals[:, -self.ipm_steps:]
        c_times_xgt = data.obj_const[:, None] * x_gt
        obj_gt = scatter(c_times_xgt, data['vals'].batch, dim=0, reduce='sum')
        return (obj_pred - obj_gt) / obj_gt

    def obj_metric(self, dataloader, model):
        model.eval()

        obj_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)
            obj_gap.append(np.abs(self.get_obj_metric(data, vals, hard_non_negative=True).detach().cpu().numpy()))

        return np.concatenate(obj_gap, axis=0)

    def constraint_metric(self, dataloader, model):
        """
        minimize ||Ax - b||^p in case of equality constraints
         ||relu(Ax - b)||^p in case of inequality

        :param dataloader:
        :param model:
        :return:
        """
        model.eval()

        cons_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)
            cons_gap.append(np.abs(self.get_constraint_violation(vals, data).detach().cpu().numpy()))

        return np.concatenate(cons_gap, axis=0)

    @torch.no_grad()
    def eval_metrics(self, dataloader, model):
        """
        both obj and constraint gap

        :param dataloader:
        :param model:
        :return:
        """
        model.eval()

        cons_gap = []
        obj_gap = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            vals, _ = model(data)
            cons_gap.append(np.abs(self.get_constraint_violation(vals, data).detach().cpu().numpy()))
            obj_gap.append(np.abs(self.get_obj_metric(data, vals, hard_non_negative=True).detach().cpu().numpy()))

        obj_gap = np.concatenate(obj_gap, axis=0)
        cons_gap = np.concatenate(cons_gap, axis=0)
        return obj_gap, cons_gap


    @torch.no_grad()
    def eval_baseline(self, dataloader, model, T):
        model.eval()

        obj_gaps = []
        constraint_gaps = []
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            val_con_repeats = model(torch.ones(1, dtype=torch.float, device=self.device) * T,
                                    data)

            vals, cons = torch.split(val_con_repeats,
                                     torch.hstack([data.num_val_nodes.sum(),
                                                   data.num_con_nodes.sum()]).tolist(), dim=0)

            obj_gaps.append(self.get_obj_metric(data, vals[:, None], True).abs().cpu().numpy())
            constraint_gaps.append(self.get_constraint_violation(vals[:, None], data).abs().cpu().numpy())

        obj_gaps = np.concatenate(obj_gaps, axis=0).squeeze()
        constraint_gaps = np.concatenate(constraint_gaps, axis=0).squeeze()

        return obj_gaps, constraint_gaps
