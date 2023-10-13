import argparse
import logging
import os

import numpy as np
import torch
import wandb
from ml_collections import ConfigDict
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose

from data.data_preprocess import HeteroAddLaplacianEigenvectorPE, SubSample
from data.dataset import LPDataset
from data.utils import args_set_bool, collate_fn_ip
from models.hetero_gnn import TripartiteHeteroGNN
from trainer import Trainer


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    # admin
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--wandbproject', type=str, default='default')
    parser.add_argument('--wandbname', type=str, default='')
    parser.add_argument('--use_wandb', type=str, default='false')
    parser.add_argument('--model_path', required=True)

    # ipm processing
    parser.add_argument('--ipm_restarts', type=int, default=1)
    parser.add_argument('--ipm_steps', type=int, default=8)
    parser.add_argument('--upper', type=float, default=1.0)

    # training dynamics
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.)  # must
    parser.add_argument('--use_norm', type=str, default='true')  # must
    parser.add_argument('--use_res', type=str, default='false')  # does not help

    # model related
    parser.add_argument('--conv', type=str, default='genconv')
    parser.add_argument('--lappe', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=8)
    parser.add_argument('--num_pred_layers', type=int, default=2)
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='mlp layers within GENConv')
    parser.add_argument('--share_conv_weight', type=str, default='false')
    parser.add_argument('--share_lin_weight', type=str, default='false')
    parser.add_argument('--conv_sequence', type=str, default='cov')

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    args = args_set_bool(vars(args))
    args = ConfigDict(args)
    logging.basicConfig(level=logging.INFO)

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

    test_loader = DataLoader(dataset,
                            batch_size=args.batchsize,
                            shuffle=False,
                            num_workers=1,
                            collate_fn=collate_fn_ip)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # test_losses = []
    test_objgap_mean = []
    test_consgap_mean = []

    model = TripartiteHeteroGNN(conv=args.conv,
                                in_shape=2,
                                pe_dim=args.lappe,
                                hid_dim=args.hidden,
                                num_conv_layers=args.num_conv_layers,
                                num_pred_layers=args.num_pred_layers,
                                num_mlp_layers=args.num_mlp_layers,
                                dropout=args.dropout,
                                share_conv_weight=args.share_conv_weight,
                                share_lin_weight=args.share_lin_weight,
                                use_norm=args.use_norm,
                                use_res=args.use_res,
                                conv_sequence=args.conv_sequence).to(device)

    trainer = Trainer(device,
                      'primal',
                      'l2',
                      1,
                      min(args.ipm_steps, args.num_conv_layers),
                      1.,
                      loss_weight=None)

    runs = [f for f in os.listdir(args.model_path) if os.path.isdir(os.path.join(args.model_path, f)) and f.startswith('run')]
    for run in runs:
        model.load_state_dict(torch.load(os.path.join(args.model_path, run, 'best_model.pt'), map_location=device))
        model.eval()
        with torch.no_grad():
            # test_loss = trainer.eval(test_loader, model, None)
            test_gaps, test_cons_gap = trainer.eval_metrics(test_loader, model)
        # test_losses.append(test_loss)
        test_objgap_mean.append(test_gaps[:, -1].mean().item())
        test_consgap_mean.append(test_cons_gap[:, -1].mean().item())

        wandb.log({'test_objgap': test_objgap_mean[-1]})
        wandb.log({'test_consgap': test_consgap_mean[-1]})
        logging.info(
            f'test_objgap: {test_objgap_mean[-1]},'
            f'test_consgap: {test_consgap_mean[-1]}')

    logging.info(f'test_objgap_stats: {np.mean(test_objgap_mean) * 100:.5f} ± {np.std(test_objgap_mean) * 100:.5f},'
                 f'test_consgap_stats: {np.mean(test_consgap_mean):.5f} ± {np.std(test_consgap_mean):.5f}')

    wandb.log({
        'test_objgap_mean': np.mean(test_objgap_mean),
        'test_objgap_std': np.std(test_objgap_mean),
        'test_consgap_mean': np.mean(test_consgap_mean),
        'test_consgap_std': np.std(test_consgap_mean),
    })
