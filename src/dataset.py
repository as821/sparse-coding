# https://github.com/lpjiang97/sparse-coding/blob/master/src/model/ImageDataset.py
# https://www.rctn.org/bruno/sparsenet/

import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat


class NatPatchDataset(Dataset):
    def __init__(self, N:int, width:int, height:int, border:int=4, fpath:str='../../data/IMAGES.mat'):
        super(NatPatchDataset, self).__init__()
        self.N = N
        self.width = width
        self.height = height
        self.border = border
        self.fpath = fpath

        self.images = None
        self._extract_patches()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx]

    def _extract_patches(self):
        X = loadmat(self.fpath)
        X = X['IMAGES']
        img_size = X.shape[0]
        n_img = X.shape[2]
        self.images = torch.zeros((self.N * n_img, self.width, self.height))

        # generate N random patches from the dataset images (patch mean removed)
        counter = 0
        for i in range(n_img):
            img = X[:, :, i]
            for j in range(self.N):
                x = np.random.randint(self.border, img_size - self.width - self.border)
                y = np.random.randint(self.border, img_size - self.height - self.border)
                crop = torch.tensor(img[x:x+self.width, y:y+self.height])
                self.images[counter, :, :] = crop - crop.mean()
                counter += 1








