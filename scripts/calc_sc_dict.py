import sys
import os
import shutil
import argparse
import torch
import torchvision
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'src/c'))

from dictionary import get_vis_path, generate_sparse_coding_dict
from preprocessor import ImagePreprocessor


def main(args):
    if args.vis_dir != "" and os.path.exists(get_vis_path(args, "")):
        shutil.rmtree(get_vis_path(args, ""))

    # load and preprocess dataset
    dset = ImagePreprocessor(args, torchvision.datasets.CIFAR10, split="train", train_set=None)
    x, img_label = dset.generate_data(args.samples)

    # learn dictionary + sparse codes jointly
    generate_sparse_coding_dict(args, x, args.dict_sz, dset)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10'], help='dataset to use')
    parser.add_argument('--dataset-path', required=True, type=str, help='path to dataset (image datasets only)')
    parser.add_argument('--samples', default=50000, type=int, help='number of training samples to use. negative for full dataset')
    parser.add_argument('--patch-sz', default=6, type=int, help='image patch size')
    parser.add_argument('--context-sz', default=32, type=int, help='other patches within this number of pixels is considered a neighbor')
    parser.add_argument('--whiten_tol', default=1e-3, type=float, help='scaling of identity added before whitening')
    parser.add_argument('--dict-sz', default=300, type=int, help='sparse coding dictionary size. should be overcomplete, larger than input data dimension by around 10x')


    parser.add_argument('--sc_lambda', default=0.03, type=float, help='value of lambda to use in ISTA')
    parser.add_argument('--sc_lr', default=0.1, type=float, help='dictionary update learning rate')
    parser.add_argument('--sc_dict_itr', default=500, type=int, help='number of ISTA/update iterations to perform when learning SC dict')
    parser.add_argument('--sc_ista_itr', default=500, type=int, help='number of iterations to perform inside ISTA')
    parser.add_argument('--sc_ista_batch', default=512, type=int, help='number of iterations to perform inside ISTA')
    parser.add_argument('--sc_lr_warm', default=20, type=int, help='number of iterations to warmup the learning rate. set <= 0 to disable')
    parser.add_argument('--sc_ista_tol', default=0.001, type=float, help='dictionary update learning rate')
    parser.add_argument('--sc_mini', default=0, type=int, help='mini-batch size for SGD. 0 to run full-dataset GD')


    parser.add_argument('--vis-dir', default="", type=str, help='path to store visualizations to')
    parser.add_argument('--vis', action='store_true', help='flag to create visualization')
    parser.add_argument('--wandb', action='store_true', help='use weights and biases')

    args = parser.parse_args()

    with torch.no_grad():
        main(args)

