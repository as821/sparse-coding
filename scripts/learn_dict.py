import argparse
import torch
import numpy as np

from dataset import NatPatchDataset
from cinterface import fista, c_impl_available
from baseline import FISTA


def main(args):
    dset = NatPatchDataset(args.nsamples, args.patch_sz, args.patch_sz, 4, args.path)
    x = dset.images.flatten(1, 2).permute((1, 0)).contiguous().numpy()
    
    basis = torch.randn((x.shape[0], args.dict_sz))
    basis /= (basis.norm(2,0) + 1e-10)

    fista_max_iter = 10000

    for ep in range(args.epoch):
        for batch_start in tqdm(range(0, x.shape[1], args.batch_sz)):
            batch_end = batch_start + args.batch_sz if batch_start + args.batch_sz < x.shape[1] else x.shape[1]
            
            if c_impl_available():
                z = fista(x[:, batch_start:batch_end], basis.numpy(), args.alpha, fista_max_iter)
            else:
                z = FISTA(x, basis, args.alpha, fista_max_iter)

            basis += lr * (residual @ z.T) 
            basis /= (torch.linalg.vector_norm(basis, dim=0) + 1e-10)
            

    # TODO(as) store checkpoint






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="/Users/andrewstange/Desktop/research/sparse-coding/data/IMAGES.mat", type=str, help="dataset path")
    parser.add_argument('--patch_sz', default=10, type=int, help="patch size")
    parser.add_argument('--nsamples', default=2000, type=int, help="batch size")
    parser.add_argument('--dict_sz', default=128, type=int, help="dictionary size")
    parser.add_argument('--batch_sz', default=2048, type=int, help="number of samples to process at once")
    parser.add_argument('--lr', default=1e-2, type=float, help="dictionary learning rate")
    parser.add_argument('--alpha', default=5e-3, type=float, help="alpha parameter for FISTA")


    main(parser.parse_args())




