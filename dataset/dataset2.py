import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, Sampler
import numpy as np
from typing import Iterable, Iterator, Optional, Sized, Union, Callable, List, Tuple
from math import sqrt
import random


_Number = Union[float, int]


def random_roll(img, label):
    # possible problem: will it work properly if inside the dataloader?
    shifts = tuple(np.random.randint(low=0, high=img.shape, size=(2,)))
    return img.roll(shifts=shifts, dims=(0,1)), label


def identity(img, label):
    return img, label


def make_3_channels(img, label):
    return torch.tile(img, (3,1,1)), label


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


class TilingDataset(Dataset):
    image_side = 64
    image_area = image_side * image_side

    def __init__(self,
                 images: torch.Tensor,
                 labels: torch.Tensor,
                 num_tiles: int = 16,
                 transform: Optional[Union[Callable, List[Callable]]] = [random_roll,]) -> None:
        super().__init__()

        assert not self.image_side % num_tiles

        self.images = images
        self.labels = labels
        self.transform = transform
        self.num_tiles = num_tiles

        self.tile_area = self.image_area // self.num_tiles
        self.tile_side = int(sqrt(self.tile_area))
    
    def get_info(self, image: torch.Tensor) -> List[_Number]:
        info = []
        tiles = self.split_into_tiles(image, self.tile_side, self.image_side)

        for tile in tiles:
            info += [self.calculate_material(tile)]
        
        return info

    def split_into_tiles(self,
                         image: torch.Tensor,
                         tile_side: int,
                         image_side: int) -> List[torch.Tensor]:
        tiles = [
            image[i: i + tile_side, j: j + tile_side] for i in range(0, image_side, tile_side) for j in range(0, image_side, tile_side)
        ]
        
        return tiles

    def calculate_material(self, tile: torch.Tensor) -> _Number:
        return tile.sum().item() / self.tile_area
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self.images[index]
        y: torch.Tensor = self.labels[index]
        if callable(self.transform):
            x, y = self.transform(x, y)
        elif isinstance(self.transform, list): # erzatz for transfoms.Compose
            for tr in self.transform:
                x, y = tr(x, y)
        
        is_broken = y[-1].item()
        info = self.get_info(x) + [is_broken]
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        return x, torch.Tensor(info)

    def __len__(self) -> int:
        return self.images.shape[0]


class MaterialPercentDataset(Dataset):
    image_area = 64 * 64

    def __init__(self,
                 images: torch.Tensor,
                 labels: torch.Tensor,
                 transform=[random_roll,]) -> None:
        super().__init__()
        
        self.images = images
        self.labels = labels
        self.transform = transform

    def calc_percent(self, image: torch.Tensor) -> torch.Tensor:
        material_count = image.sum()
        return material_count / self.image_area

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.images[index]
        y = self.labels[index]
        if callable(self.transform):
            x, y = self.transform(x, y)
        elif isinstance(self.transform, list): # erzatz for transfoms.Compose
            for tr in self.transform:
                x, y = tr(x, y)
        
        percent = self.calc_percent(x)

        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        return x, percent.unsqueeze(-1), y


class BinarizedDataset(MaterialPercentDataset):
    def __init__(self,
                 images: torch.Tensor,
                 labels: torch.Tensor,
                 bins_number: int = 15,
                 transform=[identity,]) -> None:
        super().__init__(images, labels, transform)

        self.bins_number = bins_number

        self.linspace = torch.linspace(0, 1, steps=bins_number)
    
    def one_hot(self, scalar: _Number) -> torch.Tensor:
        _, idx = torch.min(torch.abs(self.linspace - scalar), dim=0)
        out = torch.zeros(self.bins_number)
        out[idx] = 1
        return out
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        x = self.images[index]
        y = self.labels[index]
        if callable(self.transform):
            x, y = self.transform(x, y)
        elif isinstance(self.transform, list): # erzatz for transfoms.Compose
            for tr in self.transform:
                x, y = tr(x, y)
        
        is_broken = y[-1].unsqueeze(0)
        percent = self.calc_percent(x).item()
        condition = self.one_hot(percent)
        condition = torch.cat((condition, is_broken), dim=0)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        return x, condition, y


class TripletBinsDataset(BinarizedDataset):
    '''
        bins_indexes Tensor is supposed to be ids from the extended_clean_dataset.pt
    '''
    image_area = 64 * 64

    def __init__(self,
                 images: torch.Tensor,
                 labels: torch.Tensor,
                 bins_indexes: torch.Tensor,
                 bins_number: int = 4,
                 transform=[identity,]) -> None:
        super().__init__(None, None, bins_number)

        self.images = images
        self._labels = labels
        self.transform = transform
        self.bins_indexes = bins_indexes.float()

        self.make_appropriate_labels()
    
    def make_appropriate_labels(self):
        new_labels = []

        for label in self._labels:
            bin = np.nonzero(label.numpy())[0][0]
            new_labels.append(bin)
        
        self.labels = torch.tensor(new_labels, dtype=torch.int8)
    
    def __len__(self) -> int:
        return self.images.shape[0]
    
    def make_condition(self,
                       x: torch.Tensor) -> torch.Tensor:
        percent = self.calc_percent(x).item()
        condition = self.one_hot(percent)
        return condition

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor = self.images[index]
        anchor_label = self.labels[index]
        
        positive_list = self.bins_indexes[self.bins_indexes != index]
        positive_list = positive_list[self.labels[self.bins_indexes != index] == anchor_label]

        '''
            TODO: сделать выборку из негативов, которые могут состоять из того же скелета,
            тогда они будут находиться в других бинах. Если первой опции нет, то просто 
            берем выборку из других бинов.
        '''
        negative_list = self.bins_indexes[self.bins_indexes == index]
        negative_list = negative_list[self.labels[self.bins_indexes == index] != anchor_label]
        
        if not len(negative_list):
            negative_list = self.bins_indexes[self.bins_indexes != index]
            negative_list = negative_list[self.labels[self.bins_indexes != index] != anchor_label]
        
        positive = self.images[random.choice(positive_list.long())]
        negative = self.images[random.choice(negative_list.long())]

        anchor_condition = self.make_condition(anchor)
        pos_condition = self.make_condition(positive)
        neg_condition = self.make_condition(negative)

        for tr in self.transform:
            anchor, _  = tr(anchor, None)
            positive, _ = tr(positive, None)
            negative, _ = tr(negative, None)
        
        if anchor.ndim == 2:
            anchor = anchor.unsqueeze(0)
            positive = positive.unsqueeze(0)
            negative = negative.unsqueeze(0)
        
        return (anchor, anchor_condition), (positive, pos_condition), (negative, neg_condition)


class BalancedBatchSampler(BatchSampler):
    def __init__(self,
                 labels: torch.Tensor,
                 batch_size: int,
                 num_classes: int,
                 sampler=Sampler(None),
                 drop_last: bool = True) -> None:
        super().__init__(sampler, batch_size, drop_last)

        self.labels = labels
        self.num_classes = num_classes

        self.classes_indexes = [[] for _ in range(num_classes)]

        for i, label in enumerate(self.labels):
            class_label = np.nonzero(label.numpy())[0][0]
            self.classes_indexes[class_label].append(i)
        
        self.samples_per_class = min(len(t) for t in self.classes_indexes)
    
    def __iter__(self) -> Iterator[List[int]]:
        pretty_wmn_walking_down_the_street = []

        for class_index in self.classes_indexes:
            indexes = np.random.choice(class_index, size=self.samples_per_class)
            pretty_wmn_walking_down_the_street.extend(torch.from_numpy(indexes))
        
        yield pretty_wmn_walking_down_the_street
    
    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size


if __name__ == "__main__":
    # test = torch.randn(1, 1, 64, 64)
    # transform = Resize((64, 64))
    test = torch.rand(100, 3)

    print(test[[3, 4, 5, 2, 5, 6, 2, 4 ]])
    exit()

    # t = transform(test)
    # print(t == test)
    from torchvision.utils import make_grid

    new_imgs = []
    new_labels = []

    imgs, labels = torch.load("dataset/extended_clean_dataset.pt")
    new_imgs = []
    new_labels = []
    new_indexes = []
    # print(imgs[0].sum().item())

    dataset = MaterialPercentDataset(imgs, labels)
    
    import cv2
    dilatation_size = 2
    element = cv2.getStructuringElement(cv2.MORPH_OPEN, (2 * dilatation_size + 1, 2 * dilatation_size + 1),(dilatation_size, dilatation_size))
    
    bins = 2
    bins_distribution = torch.zeros(bins)
    dataset = BinarizedDataset(imgs, labels, bins_number=bins)

    for (img, label, _) in dataset:
        if label.numpy().astype(int)[0] == 1: 
            dilatation_dst = cv2.dilate(img.numpy() ,element)
            
            new_image = torch.from_numpy(dilatation_dst)
            bins_distribution += dataset.one_hot(dataset.calc_percent(new_image).item())


        bins_distribution += dataset.one_hot(dataset.calc_percent(img).item())
    
    
    bins_distribution = bins_distribution.numpy()

    from matplotlib import pyplot as plt

    # bins_distribution = [14, 0, 721, 778, 1182, 1862, 1950, 1817, 1592, 1310, 1025, 715, 490, 498, 76]
    print(bins_distribution)

    plt.scatter(range(bins), bins_distribution)
    plt.grid(True)
    plt.show()

    # item = dataset[110]
    # print(item[1], item[2])


