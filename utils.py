from typing import Tuple
import os
import random
import numpy as np
import torch
from torch import nn
from gans.gan_config import tox_config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def train_test_split(images: torch.Tensor,
                     labels: torch.Tensor,
                     test_size: float) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
    data_size = len(images)
    test_size = int(data_size * test_size)

    permutation = torch.randperm(data_size)

    train = (images[permutation[test_size:]], labels[permutation[test_size:]])
    test = (images[permutation[:test_size]], labels[permutation[:test_size]])
    return train, test


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    a = np.array([[1,0,0,1],[1,0,1,1],[1,0,1,1],[1,0,1,0]])
    plt.imshow(a)
    plt.show()
    ah = np.hstack([a,a])
    plt.imshow(ah)
    plt.show()
    av = np.vstack([a,a])
    plt.imshow(av)
    plt.show()

    plt.imshow(av[av.shape[0]//2], av[av.shape[0]//2-1])
    plt.show()

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    loss_v = mean_squared_error(av[av.shape[0]//2], av[av.shape[0]//2-1])
    print("loss_v:", loss_v)

    loss_h = mean_squared_error(ah[:, ah.shape[1]//2], ah[:, ah.shape[1]//2-1])
    print("loss_h:", loss_h)
