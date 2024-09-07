import argparse
import torch
from tqdm import tqdm
import numpy as np
import time
import wandb

import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'src/c'))


from dataset import NatPatchDataset
from cinterface import fista, c_impl_available
from baseline import FISTA


def plot_rf(rf, out_dim, M):
    # rf = rf.reshape(out_dim, -1)
    # # normalize
    rf /= rf.abs().max(dim=1, keepdims=True)[0]
    rf = rf.reshape(out_dim, M, M)

    # plotting
    n = 10 # int(np.ceil(np.sqrt(rf.shape[0])))
    fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
    fig.set_size_inches(10, 10)
    for i in range(n * n):
        axes.flat[i].imshow(rf[i], cmap='gray', vmin=-1, vmax=1)
        axes.flat[i].set_xticks([])
        axes.flat[i].set_yticks([])
        axes.flat[i].set_aspect('equal')
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

    dset = NatPatchDataset(args.nsamples, args.patch_sz, args.patch_sz, 4, args.path)
    x = dset.images.flatten(1, 2).permute((1, 0)).contiguous().numpy()
    
    basis = torch.randn((x.shape[0], args.dict_sz), requires_grad=True)
    with torch.no_grad():
        basis /= (basis.norm(2,0) + 1e-10)

    optim = torch.optim.SGD([{'params': basis, "lr": args.lr}])

    fista_max_iter = 10000

    if args.wandb:
        wandb.init(config=args, project="smt_sc_dict")

    for ep in range(args.epoch):
        running_loss = 0
        vis_dict = {}
        for batch_start in tqdm(range(0, x.shape[1], args.batch_sz)):
            batch_end = batch_start + args.batch_sz if batch_start + args.batch_sz < x.shape[1] else x.shape[1]
            x_batch = torch.tensor(x[:, batch_start:batch_end], requires_grad=True)
            
            with torch.no_grad():
                if c_impl_available():
                    z_np = fista(x_batch.detach().numpy().contiguous(), basis.detach().numpy(), args.alpha, fista_max_iter)
                else:
                    z_np = FISTA(x_batch.detach(), basis.detach(), args.alpha, fista_max_iter, tqdm_disable=True).numpy()            
            z = torch.tensor(z_np, requires_grad=True)

            pred = basis @ z
            loss = ((x_batch - pred)**2).sum()
            loss.backward()
            optim.step()
            optim.zero_grad()
            with torch.no_grad():
                basis /= (torch.linalg.vector_norm(basis, dim=0) + 1e-10)

            running_loss += np.abs(loss.detach().numpy()).sum()


        print(f"Epoch {ep}: {running_loss}")
        vis_dict["loss"] = running_loss
        if ep % 5 == 0:
            fig = plot_rf(basis.detach().T, args.dict_sz, args.patch_sz)
            if args.wandb:
                vis_dict["dict"] = wandb.Image(plt)
            else:
                plt.show()

        if ep % 10 == 0 and args.ckpt_path != "":
            torch.save(basis, args.ckpt_path + f"/{timestamp}/ckpt_{ep}.pt")

        if args.wandb:
            wandb.log(vis_dict, step=ep)

    if args.ckpt_path != "":
        torch.save(basis, args.ckpt_path + f"/{timestamp}/final.pt")
    if args.wandb:
        wandb.finish()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="/Users/andrewstange/Desktop/research/sparse-coding/data/IMAGES.mat", type=str, help="dataset path")
    parser.add_argument('--ckpt-path', default="", type=str, help="checkpoint directory")

    parser.add_argument('--patch_sz', default=10, type=int, help="patch size")
    parser.add_argument('--nsamples', default=2000, type=int, help="batch size")
    parser.add_argument('--epoch', default=100, type=int, help="number of epochs")
    parser.add_argument('--dict_sz', default=128, type=int, help="dictionary size")
    parser.add_argument('--batch_sz', default=256, type=int, help="number of samples to process at once")
    parser.add_argument('--lr', default=1e-2, type=float, help="dictionary learning rate")
    parser.add_argument('--alpha', default=5e-3, type=float, help="alpha parameter for FISTA")
    parser.add_argument('--wandb', action="store_true")


    main(parser.parse_args())




