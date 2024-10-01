import argparse
import torch
from tqdm import tqdm
import numpy as np
import time
import wandb
import matplotlib
matplotlib.use('Agg')       # non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math


import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'src/c'))


from dataset import NatPatchDataset, CIFAR10RandomPatch
from cinterface import multi_gpu_cu_fista, fista, c_impl_available
from baseline import FISTA
from plot import plot_color, plot_patch_recon

def get_alpha(step, args, dataloader):
    if args.alpha >= 0:
        return args.alpha
    if step < args.alpha_constant_steps:
        return args.alpha_initial
    else:
        progress = (step - args.alpha_constant_steps) / (args.epoch * len(dataloader) - args.alpha_constant_steps)
        return args.alpha_initial + (args.alpha_final - args.alpha_initial) * progress

def restart_dead_dict_el(args, basis, z, hist):
    curr_inactive = (z != 0).to(int).sum(dim=0) == 0
    if hist is not None:
        # currently inactive and previously inactive
        high = 255
        low = -255
        need_restart = torch.logical_and(curr_inactive, hist)
        n_restart = need_restart.sum().item()
        if n_restart > 0:
            # NOTE: caller needs to normalize for us. Do not clear curr_inactive for these restarted elements since 
            # want to restart again in case they still have no activations
            basis[:, need_restart] = torch.rand_like(basis[:, need_restart]) * (high - low) + low
            print(f"Restarted {n_restart} dict. elements")
    return basis, curr_inactive

def main(args):
    timestamp = time.time()
    if args.ckpt_path != "":
        assert os.path.exists(args.ckpt_path)
        assert not os.path.exists(args.ckpt_path + f"/{timestamp}")
        os.mkdir(args.ckpt_path + f"/{timestamp}")

    if args.wandb:
        wandb.init(config=args, project="smt_sc_dict")


    device = torch.device("cuda" if torch.cuda.is_available() and not c_impl_available() else "cpu")
    fista_max_iter = 10000
    
    basis_shape = args.patch_sz ** 2
    if args.dataset == "nat":
        dset = NatPatchDataset(args.nsamples, args.patch_sz, args.patch_sz, fpath=args.path)
    elif args.dataset == "cifar10":
        dset = CIFAR10RandomPatch(args.nsamples, args.patch_sz, args.patch_sz, fpath=args.path)
        basis_shape *= 3
    dataloader = DataLoader(dset, batch_size=args.batch_sz, drop_last=True, num_workers=4)

    basis = nn.Linear(args.dict_sz, basis_shape, bias=False, dtype=torch.float32).to(device)
    if args.init_unif != 0:
        nn.init.uniform_(basis.weight, a=-args.init_unif, b=args.init_unif, generator=None)
    with torch.no_grad():
        basis.weight.data = F.normalize(basis.weight.data, dim=0)
    optim = torch.optim.Adam([{'params': basis.weight, "lr": args.lr}])
    if args.dropout > 0:
        dropout = nn.Dropout(p=args.dropout)
    
    step_cnt = 0
    history = None
    for e in range(args.epoch):
        vis_dict = {}
        running_loss = 0
        c = 0
        for img_batch in tqdm(dataloader, desc='training', total=len(dataloader)):
            s = time.time()
            img_batch = img_batch.to(device)
            with torch.no_grad():
                alpha = get_alpha(step_cnt, args, dataloader)
                step_cnt += 1
                if c_impl_available():
                    z, n_iter, _ = multi_gpu_cu_fista(img_batch, basis.weight, alpha, fista_max_iter, args.fista_conv, args.fista_lr)
                
                    if args.cuda_profile:
                        sys.exit(1)

                    # TODO(as): debugging
                    # foo = basis.weight.clone()
                    # z_gt, n_iter_gt = FISTA(img_batch.clone().to("cuda"), foo.to("cuda"), args.alpha, fista_max_iter, args.fista_conv, "cuda", lr=args.fista_lr)
                    # assert n_iter == n_iter_gt, f"{n_iter} != {n_iter_gt}"
                    # assert np.all(np.abs(z_gt.cpu().numpy() - z.numpy()) < 0.1), f"{np.abs(z_gt.cpu().numpy() - z.numpy()).max()} >= 0.1"
                else:
                    z, n_iter = FISTA(img_batch, basis.weight, alpha, fista_max_iter, args.fista_conv, device, lr=args.fista_lr)
                vis_dict['fista_niter'] = n_iter
                vis_dict['alpha'] = alpha
            t1 = time.time()
            if args.dropout > 0:
                z = dropout(z)
            pred = basis(z)
            t2 = time.time()
            loss = ((img_batch - pred) ** 2).sum()
            running_loss += loss.item()
            t3 = time.time()
            
            loss.backward()
            optim.step()
            basis.zero_grad()
            t4 = time.time()
            
            with torch.no_grad():
                if args.restart_freq > 0 and step_cnt % args.restart_freq == 0 and step_cnt > 1:
                    basis.weight.data, history = restart_dead_dict_el(args, basis.weight.data, z, history)

                basis.weight.data = F.normalize(basis.weight.data, dim=0)

            # print(f"{t1 - s} {t2 - t1} {t3 - t2} {t4 - t3}")
            c += 1

            log_freq = 5
            if (step_cnt % log_freq == 0 or c == 1) and args.wandb:
                vis_dict['loss'] = running_loss / c

                n_activations_per_sample = (z != 0).to(int).sum(dim=1)
                vis_dict['max_sample_active'] = n_activations_per_sample.max()
                vis_dict['min_sample_active'] = n_activations_per_sample.min()
                vis_dict['mean_sample_active'] = n_activations_per_sample.float().mean()
                vis_dict['n_zero_sample_active'] = (n_activations_per_sample == 0).to(int).sum()

                n_activations_per_dict = (z != 0).to(int).sum(dim=0)
                vis_dict['max_dict_active'] = n_activations_per_dict.max()
                vis_dict['min_dict_active'] = n_activations_per_dict.min()
                vis_dict['mean_dict_active'] = n_activations_per_dict.float().mean()
                vis_dict['n_zero_dict_active'] = (n_activations_per_dict == 0).to(int).sum()

                step_div = math.floor(step_cnt / log_freq)
                if step_div % 5 == 4:
                    assert args.dataset == "cifar10"
                    fig = plot_color(basis.weight.T.reshape(args.dict_sz, args.patch_sz, args.patch_sz, 3).cpu().data.numpy(), args.dict_sz, args.patch_sz)
                    vis_dict["dict"] = wandb.Image(plt)
                    plt.close()

                    fig = plot_patch_recon(args, basis.weight.reshape(args.patch_sz, args.patch_sz, 3, args.dict_sz).cpu().data, img_batch.cpu(), z)
                    vis_dict["recon"] = wandb.Image(plt)
                    plt.close()

                wandb.log(vis_dict, step=step_cnt)
                

        if e % 1 == 0 and args.ckpt_path != "":
            torch.save(basis, args.ckpt_path + f"/{timestamp}/ckpt_{e}.pt")
    
    if args.ckpt_path != "":
        torch.save(basis, args.ckpt_path + f"/{timestamp}/final.pt")
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="/Users/andrewstange/Desktop/research/sparse-coding/data/cifar10", type=str, help="dataset path")
    parser.add_argument('--ckpt-path', default="", type=str, help="checkpoint directory")
    parser.add_argument('--nsamples', default=20, type=int, help="batch size")
    parser.add_argument('--dict_sz', default=512, type=int, help="dictionary size")
    parser.add_argument('--patch_sz', default=10, type=int, help="patch size")
    parser.add_argument('--epoch', default=50, type=int, help="number of epochs")
    parser.add_argument('--lr', default=1e-2, type=float, help="dictionary learning rate")
    parser.add_argument('--fista_conv', default=0.01, type=float, help="convergence threshold for FISTA")
    parser.add_argument('--fista_lr', default=0.001, type=float, help="learning rate for FISTA")
    parser.add_argument('--dataset', default='cifar10', choices=['nat', 'cifar10'], help='dataset to use')
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--cuda_profile', action="store_true")
    parser.add_argument('--batch_sz', default=2048, type=int, help="batch size")
    parser.add_argument('--init_unif', default=0, type=float, help="uniform initialization range")

    parser.add_argument('--alpha', default=0.0005, type=float, help="constant alpha parameter for FISTA, -1 to use scheduled alpha instead")
    parser.add_argument('--alpha_initial', type=float, default=0.001, help='initial alpha value')
    parser.add_argument('--alpha_final', type=float, default=0.05, help='final alpha value')
    parser.add_argument('--alpha_constant_steps', type=int, default=150, help='# steps to keep alpha constant')
    parser.add_argument('--dropout', default=0, type=float, help="dropout rate, <=0 to disable")
    parser.add_argument('--restart_freq', type=int, default=0, help='frequency at which to check for dead dictionary elements')

    main(parser.parse_args())
