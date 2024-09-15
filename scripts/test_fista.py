import argparse
import torch
import numpy as np

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'src/c'))

from dataset import NatPatchDataset
from baseline import FISTA
from cinterface import fista, cu_fista


def main(args):
    dset = NatPatchDataset(args.nsamples, args.patch_sz, args.patch_sz, args.path, 4)
    x = dset.images.flatten(1, 2)
    

    # x = x[:1][:5]
    # x = x[:3, :2].contiguous()

    basis = torch.randn((x.shape[1], args.dict_sz))
    basis = basis / (basis.norm(2,0) + 1e-10)


    lr = 0.01
    alpha = 0.01
    niter = 10000
    thresh = 0.01
    
    n_test = 10
    res = np.zeros(shape=(n_test,))
    for idx in range(n_test):
        c_res, n_iter, T = cu_fista(x, basis, alpha, niter, thresh, lr)
        res[idx] = T
        print("\n\n")
    # fista(x, basis, alpha, niter, thresh, lr)
    # print("\n\n")


    print(f"\n\nFINAL: {res.sum():.4f} ({res.mean():.4f} {res.std():.4f} {res.max():.4f} {res.min():.4f})")

    if args.comparison:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        python_res = FISTA(x.to(device), basis.to(device), alpha, niter, converge_thresh=thresh, lr=lr, device=device, tqdm_disable=False)[0].cpu()
        diff = np.abs(python_res.numpy() - c_res.numpy())
        print(f"{diff.max():.4f} {diff.mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="/Users/andrewstange/Desktop/research/sparse-coding/data/IMAGES.mat", type=str, help="dataset path")
    parser.add_argument('--patch_sz', default=10, type=int, help="patch size")
    parser.add_argument('--nsamples', default=20000, type=int, help="batch size")
    parser.add_argument('--dict_sz', default=128, type=int, help="dictionary size")
    parser.add_argument('--comparison', action="store_true", help="check correctness of C implementation")

    main(parser.parse_args())




