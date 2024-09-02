import argparse
import torch
from tqdm import tqdm
import numpy as np
import time
import wandb

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'src/c'))


from dataset import NatPatchDataset
from cinterface import fista, c_impl_available
from baseline import FISTA


def main(args):
    timestamp = time.time()
    if args.ckpt_path != "":
        assert os.path.exists(args.ckpt_path)
        assert not os.path.exists(args.ckpt_path + f"/{timestamp}")
        os.mkdir(args.ckpt_path + f"/{timestamp}")

    dset = NatPatchDataset(args.nsamples, args.patch_sz, args.patch_sz, 4, args.path)
    x = dset.images.flatten(1, 2).permute((1, 0)).contiguous().numpy()
    
    basis = torch.randn((x.shape[0], args.dict_sz))
    basis /= (basis.norm(2,0) + 1e-10)

    fista_max_iter = 10000

    if args.wandb:
        wandb.init(config=args, project="smt_sc_dict")

    for ep in range(args.epoch):
        running_loss = 0
        vis_dict = {}
        for batch_start in tqdm(range(0, x.shape[1], args.batch_sz)):
            batch_end = batch_start + args.batch_sz if batch_start + args.batch_sz < x.shape[1] else x.shape[1]
            
            x_batch = x[:, batch_start:batch_end]
            if c_impl_available():
                z = fista(x_batch, basis.numpy(), args.alpha, fista_max_iter)
            else:
                z = FISTA(torch.from_numpy(x_batch), basis, args.alpha, fista_max_iter, tqdm_disable=True).numpy()

            residual = x_batch - basis.numpy() @ z
            running_loss += np.abs(residual).sum()
            basis += args.lr * (residual @ z.T) 
            basis /= (torch.linalg.vector_norm(basis, dim=0) + 1e-10)

        print(f"Epoch {ep}: {running_loss}")
        vis_dict["loss"] = running_loss
        if ep % 5 == 0:
            # TODO(as): visualize dict + save checkpoint
            pass

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
    parser.add_argument('--ckpt-path', default="/Users/andrewstange/Desktop/research/sparse-coding/ckpts", type=str, help="checkpoint directory")

    parser.add_argument('--patch_sz', default=10, type=int, help="patch size")
    parser.add_argument('--nsamples', default=2000, type=int, help="batch size")
    parser.add_argument('--epoch', default=100, type=int, help="number of epochs")
    parser.add_argument('--dict_sz', default=128, type=int, help="dictionary size")
    parser.add_argument('--batch_sz', default=256, type=int, help="number of samples to process at once")
    parser.add_argument('--lr', default=1e-2, type=float, help="dictionary learning rate")
    parser.add_argument('--alpha', default=5e-3, type=float, help="alpha parameter for FISTA")
    parser.add_argument('--wandb', action="store_true")


    main(parser.parse_args())




