import os
import random
import numpy as np
import torch
import cv2
from dataset2 import BinarizedDataset, TripletBinsDataset, random_roll

seed = 42

random.seed(seed)
os.environ['PYTHONASSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def make_triplet_dataset(dataset: BinarizedDataset) -> None:
    bins_number = dataset.bins_number

    bin_images = [[] for _ in range(bins_number)]
    bin_labels = [[] for _ in range(bins_number)]
    bin_indexes = [[] for _ in range(bins_number)]

    dilatation_size = 3
    element = cv2.getStructuringElement(cv2.MORPH_OPEN, (2 * dilatation_size + 1, 2 * dilatation_size + 1),(dilatation_size, dilatation_size))

    for i, (img, label, _) in enumerate(dataset):
        label = dataset.one_hot(dataset.calc_percent(img).item())

        bin = np.nonzero(label.numpy())[0][0]

        bin_images[bin].append(img)
        bin_labels[bin].append(label.unsqueeze(0))
        bin_indexes[bin].append(torch.tensor([i], dtype=torch.int8))

        if bin in [1, 2]:

            dilatation_dst = cv2.dilate(img.numpy() ,element)
            new_image = torch.from_numpy(dilatation_dst)
            
            new_label = dataset.one_hot(dataset.calc_percent(new_image).item())

            new_bin = np.nonzero(new_label.numpy())[0][0]

            bin_images[new_bin].append(new_image)
            bin_labels[new_bin].append(new_label.unsqueeze(0))
            bin_indexes[new_bin].append(torch.tensor([i], dtype=torch.int8))
    
    # augment fisrt bin and do nothing with other ones.
    tmp_images, tmp_labels, tmp_indexes = [], [], []
    for img, label, index in zip(bin_images[0], bin_labels[0], bin_indexes[0]):
        new_i, new_l = random_roll(img.squeeze(0), label)
        pass
        new_i = new_i.unsqueeze(0)
        tmp_images.append(new_i)
        tmp_labels.append(new_l)
        tmp_indexes.append(index)
    
    bin_images[0].extend(tmp_images)
    bin_labels[0].extend(tmp_labels)
    bin_indexes[0].extend(tmp_indexes)

    print('\t'.join(str(len(t)) for t in bin_images))

    for i in range(bins_number):
        assert len(bin_images[i]) == len(bin_labels[i]) == len(bin_indexes[i]), f"bin {i}"
    
    new_images = []
    new_labels = []
    new_indexes = []

    for (img_list, labels_list, indexes_list) in zip(bin_images, bin_labels, bin_indexes):
        new_images.append(torch.cat(img_list, dim=0))
        new_labels.append(torch.cat(labels_list, dim=0))
        new_indexes.append(torch.cat(indexes_list, dim=0))
    
    images = torch.cat(new_images, dim=0)
    labels = torch.cat(new_labels, dim=0)
    indexes = torch.cat(new_indexes, dim=0)

    print(images.shape, labels.shape, indexes.shape)

    torch.save((images, labels, indexes), f"triplet_extended_dataset{bins_number}.pt")




if __name__ == "__main__":
    # imgs, labels = torch.load("dataset/extended_clean_dataset.pt")

    # dataset = BinarizedDataset(imgs, labels, bins_number=4)
    # make_triplet_dataset(dataset)

    # dataset = TripletBinsDataset(imgs, labels, indexes)
    # label = dataset.labels[0]

    # dataset[5]
    # # print(dataset.bins_indexes[dataset.labels[dataset.bins_indexes == 3] == label])

    # # for i in range(len(dataset)):
    #     # anchor, pos, neg = dataset[i]
    #     # print(anchor.shape, pos.shape, neg.shape)
    
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, 16)

    # for (anchor, pos, neg, label) in dataloader:
    #     print(anchor.shape, pos.shape, neg.shape)
    #     print(label.shape)
    #     break
    # print(np.nonzero([1.0, 0.0, 0.0])[0][0])

    from dataset2 import TripletBinsDataset, BalancedBatchSampler
    from torch.utils.data import DataLoader

    imgs, labels, indexes = torch.load("dataset/triplet_extended_dataset4.pt")
    dataset = TripletBinsDataset(imgs, labels, indexes)
    sampler = BalancedBatchSampler(labels, batch_size=64, num_classes=4)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    for i, (anchor_tuple, pos_tuple, neg_tuple) in enumerate(dataloader):
        print(anchor_tuple[1].sum(dim=0))
        print(pos_tuple[1].sum(dim=0))
        print(neg_tuple[1].sum(dim=0))
        print()


