# https://github.com/lpjiang97/sparse-coding/blob/master/src/model/ImageDataset.py
# https://www.rctn.org/bruno/sparsenet/

import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

class RandomPatchDataset(Dataset):
    def __init__(self, N:int, width:int, height:int, fpath:str, border:int=4):
        super(RandomPatchDataset, self).__init__()
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
    
    def _get_full_raw_dataset(self):
        return None

    def _extract_patches(self):
        X = self._get_full_raw_dataset()
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
        # self.images = self.images.reshape(self.images.shape[0], -1)

class NatPatchDataset(RandomPatchDataset):
    def __init__(self, N:int, width:int, height:int, fpath:str, border:int=4):
        super(NatPatchDataset, self).__init__(N, width, height, fpath, border)

    def _get_full_raw_dataset(self):

        X = loadmat(self.fpath)
        X = X['IMAGES']
        return X

class CIFAR10RandomPatch(RandomPatchDataset):
    def __init__(self, N:int, width:int, height:int, fpath:str, border:int=4):
        self.dataset = torchvision.datasets.CIFAR10(root=fpath, train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
        super(CIFAR10RandomPatch, self).__init__(N, width, height, fpath, border)

    def _get_full_raw_dataset(self):
        return self.dataset.data

    def _extract_patches(self):
        X = self._get_full_raw_dataset()
        img_size = X.shape[1]
        n_img = X.shape[0]
        self.images = torch.zeros((self.N, self.width * self.height * X.shape[-1]))

        # generate N random patches from the dataset images (patch mean removed)
        counter = 0

        for i in tqdm(range(self.N)):
            img = X[i % n_img]
            x = np.random.randint(self.border, img_size - self.width - self.border)
            y = np.random.randint(self.border, img_size - self.height - self.border)
            crop = img[x:x+self.width, y:y+self.height]
            crop = crop.flatten()
            self.images[counter] = torch.from_numpy(crop - crop.mean())
            counter += 1






