import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class StructuresDatasetTensor(Dataset):
    def __init__(self, imgs, labels, transform=None):
        assert imgs.shape[0] == labels.shape[0]
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.imgs[index]
        y = self.labels[index]
        if callable(self.transform):
            x, y = self.transform(x, y)
        elif isinstance(self.transform, list): # erzatz for transfoms.Compose
            for tr in self.transform:
                x, y = tr(x, y)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        return x, y

    def __len__(self):
        return self.imgs.shape[0]


def random_roll(img, label):
    # possible problem: will it work properly if inside the dataloader?
    shifts = tuple(np.random.randint(low=0, high=img.shape, size=(2,)))
    return img.roll(shifts=shifts, dims=(0,1)), label


def make_3_channels(img, label):
    return torch.tile(img, (3,1,1)), label