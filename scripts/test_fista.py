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
from cinterface import fista


def main(args):
    dset = NatPatchDataset(args.nsamples, args.patch_sz, args.patch_sz, 4, args.path)
    x = dset.images.flatten(1, 2)
    basis = torch.randn((x.shape[1], args.dict_sz))
    basis = basis / (basis.norm(2,0) + 1e-10)


    c_res = fista(x, basis, 0.01, 1000)

    if args.comparison:
        python_res = FISTA(x, basis, 0.01, 1000)
        diff = np.abs(python_res.numpy() - c_res.numpy())
        print(f"{diff.max():.4f} {diff.mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="/Users/andrewstange/Desktop/research/sparse-coding/data/IMAGES.mat", type=str, help="dataset path")
    parser.add_argument('--patch_sz', default=10, type=int, help="patch size")
    parser.add_argument('--nsamples', default=2000, type=int, help="batch size")
    parser.add_argument('--dict_sz', default=128, type=int, help="dictionary size")
    parser.add_argument('--comparison', action="store_true", help="check correctness of C implementation")

    main(parser.parse_args())




