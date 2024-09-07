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


from dataset import NatPatchDataset
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

    basis = nn.Linear(args.n_neuron, args.size ** 2, bias=False).to(device)
    with torch.no_grad():
        basis.weight.data = F.normalize(basis.weight.data, dim=0)
    dataloader = DataLoader(NatPatchDataset(args.batch_size, args.size, args.size, fpath=args.path), batch_size=256)
    optim = torch.optim.SGD([{'params': basis.weight, "lr": args.learning_rate}])
    
    for e in range(args.epoch):
        running_loss = 0
        c = 0
        for img_batch in tqdm(dataloader, desc='training', total=len(dataloader)):
            img_batch = img_batch.reshape(img_batch.shape[0], -1).to(device)
            with torch.no_grad():            
                if c_impl_available():
                    assert img_batch.shape[0] % 8 == 0
                    z_np = fista(img_batch.T.contiguous().numpy(), basis.weight.numpy(), args.reg, fista_max_iter).T
                    z = torch.from_numpy(z_np)
                else:
                    z = FISTA(img_batch, basis.weight, args.reg, args.r_learning_rate, fista_max_iter, 0.01, device)
            
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
        if e % 5 == 4 and args.wandb:
            # plotting
            fig = plot_rf(basis.weight.T.reshape(args.n_neuron, args.size, args.size).cpu().data.numpy(), args.n_neuron, args.size)
            vis_dict["dict"] = wandb.Image(plt)

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
    # parser = argparse.ArgumentParser(description="Template")
    parser.add_argument('-N', '--batch_size', default=2000, type=int, help="Batch size")
    parser.add_argument('-K', '--n_neuron', default=400, type=int, help="The number of neurons")
    parser.add_argument('-M', '--size', default=10, type=int, help="The size of receptive field")
    parser.add_argument('-e', '--epoch', default=100, type=int, help="Number of Epochs")
    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help="Learning rate")
    parser.add_argument('-rlr', '--r_learning_rate', default=1e-2, type=float, help="Learning rate for ISTA")
    parser.add_argument('-lmda', '--reg', default=5e-3, type=float, help="LSTM hidden size")



    parser.add_argument('--path', default="/Users/andrewstange/Desktop/research/sparse-coding/data/IMAGES.mat", type=str, help="dataset path")
    parser.add_argument('--ckpt-path', default="", type=str, help="checkpoint directory")
    parser.add_argument('--wandb', action="store_true")

    # parser.add_argument('--patch_sz', default=10, type=int, help="patch size")
    # parser.add_argument('--nsamples', default=2000, type=int, help="batch size")
    # parser.add_argument('--epoch', default=100, type=int, help="number of epochs")
    # parser.add_argument('--dict_sz', default=128, type=int, help="dictionary size")
    # parser.add_argument('--batch_sz', default=256, type=int, help="number of samples to process at once")
    # parser.add_argument('--lr', default=1e-2, type=float, help="dictionary learning rate")
    # parser.add_argument('--alpha', default=5e-3, type=float, help="alpha parameter for FISTA")
    


    main(parser.parse_args())
