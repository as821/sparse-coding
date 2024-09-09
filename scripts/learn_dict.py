import argparse
import torch
from tqdm import tqdm
import numpy as np
import time
import wandb
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'src/c'))


from dataset import NatPatchDataset, CIFAR10RandomPatch
from cinterface import fista, c_impl_available
from baseline import FISTA


def plot_rf(rf, out_dim, M):
    rf = rf.reshape(out_dim, -1)
    rf = rf.T / np.abs(rf).max(axis=1)
    rf = rf.T
    rf = rf.reshape(out_dim, M, M)
    n = int(np.ceil(np.sqrt(rf.shape[0])))
    fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
    fig.set_size_inches(10, 10)
    for i in range(rf.shape[0]):
        ax = axes[i // n][i % n]
        ax.imshow(rf[i], cmap='gray', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    for j in range(rf.shape[0], n * n):
        ax = axes[j // n][j % n]
        ax.imshow(np.ones_like(rf[0]) * -1, cmap='gray', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    return fig

def plot_color(rf, out_dim, M):
    
    normalized = np.zeros_like(rf)
    for i in range(3):  # For each color channel
        channel = rf[:, :,:,i]
        normalized[:, :,:,i] = (channel - channel.min()) / (channel.max() - channel.min())
    
    
    
    # rf = rf.reshape(out_dim, -1)
    # rf = rf.T / np.abs(rf).max(axis=1)
    # rf = rf.T
    # rf = rf.reshape(out_dim, M, M, 3)

    n = 10 #int(np.ceil(np.sqrt(normalized.shape[0])))
    fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
    fig.set_size_inches(10, 10)
    for i in range(min(n * n, normalized.shape[0])):
        ax = axes[i // n][i % n]
        ax.imshow(normalized[i]) #, cmap='gray', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    if n * n > normalized.shape[0]:
        for j in range(normalized.shape[0], n * n):
            ax = axes[j // n][j % n]
            ax.imshow(np.ones_like(normalized[0]) * -1, cmap='gray', vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    return fig



def main(args):
    timestamp = time.time()
    if args.ckpt_path != "":
        assert os.path.exists(args.ckpt_path)
        assert not os.path.exists(args.ckpt_path + f"/{timestamp}")
        os.mkdir(args.ckpt_path + f"/{timestamp}")

    if args.wandb:
        wandb.init(config={}, project="smt_sc_dict")


    device = torch.device("cuda" if torch.cuda.is_available() and not c_impl_available() else "cpu")
    fista_max_iter = 10000
    
    basis_shape = args.patch_sz ** 2
    if args.dataset == "nat":
        dset = NatPatchDataset(args.nsamples, args.patch_sz, args.patch_sz, fpath=args.path)
    elif args.dataset == "cifar10":
        dset = CIFAR10RandomPatch(args.nsamples, args.patch_sz, args.patch_sz, fpath=args.path)
        basis_shape *= 3
    dataloader = DataLoader(dset, batch_size=args.batch_sz)



    basis = nn.Linear(args.dict_sz, basis_shape, bias=False, dtype=torch.float32).to(device)
    with torch.no_grad():
        basis.weight.data = F.normalize(basis.weight.data, dim=0)
    optim = torch.optim.SGD([{'params': basis.weight, "lr": args.lr}])
    
    for e in range(args.epoch):
        running_loss = 0
        c = 0
        for img_batch in tqdm(dataloader, desc='training', total=len(dataloader)):
            img_batch = img_batch.to(device)
            with torch.no_grad():            
                if c_impl_available():
                    assert img_batch.shape[0] % 8 == 0
                    z = fista(img_batch, basis.weight, args.alpha, fista_max_iter)
                else:
                    z = FISTA(img_batch, basis.weight, args.alpha, fista_max_iter, 0.01, device)
            
            pred = basis(z)

            loss = ((img_batch - pred) ** 2).sum()
            running_loss += loss.item()
            loss.backward()
            optim.step()
            basis.zero_grad()

            with torch.no_grad():
                basis.weight.data = F.normalize(basis.weight.data, dim=0)

            c += 1

        vis_dict = {}
        vis_dict['loss'] = running_loss / c

        n_activations_per_sample = (pred != 0).to(int).sum(dim=1)
        vis_dict['max_sample_active'] = n_activations_per_sample.max()
        vis_dict['min_sample_active'] = n_activations_per_sample.min()
        vis_dict['mean_sample_active'] = n_activations_per_sample.mean()

        if e % 5 == 4 and args.wandb:
            # plotting
            if args.dataset == "cifar10":
                fig = plot_color(basis.weight.T.reshape(args.dict_sz, args.patch_sz, args.patch_sz, 3).cpu().data.numpy(), args.dict_sz, args.patch_sz)
            else:
                fig = plot_rf(basis.weight.T.reshape(args.dict_sz, args.patch_sz, args.patch_sz).cpu().data.numpy(), args.dict_sz, args.patch_sz)
            vis_dict["dict"] = wandb.Image(plt)
            plt.close()

        if args.wandb:
            wandb.log(vis_dict, step=e)
        if e % 10 == 9 and args.ckpt_path != "":
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
    parser.add_argument('--epoch', default=100, type=int, help="number of epochs")
    parser.add_argument('--lr', default=1e-2, type=float, help="dictionary learning rate")
    parser.add_argument('--alpha', default=5e-3, type=float, help="alpha parameter for FISTA")
    parser.add_argument('--dataset', default='cifar10', choices=['nat', 'cifar10'], help='dataset to use')
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--batch_sz', default=2048, type=int, help="batch size")


    main(parser.parse_args())
