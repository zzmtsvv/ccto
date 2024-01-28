from matplotlib import pyplot as plt
import random
import torch
from torchvision.transforms import Resize
from dataset2 import StructuresDatasetTensor, random_roll, make_3_channels


if __name__ == "__main__":

    imgs, labels = torch.load('dataset/clean_dataset.pt')
    device = "cpu"
    ds = StructuresDatasetTensor(imgs=imgs.to(device),
                                 labels=labels.to(device),)
    #                                 #transform=[random_roll,]
    #                             ) #, make_3_channels
    
    # for i in range(100):
    #     if labels[i][-1] == 1:
    #         continue
    #     print(labels[i][-1])
    # print(len([l for l in labels if l[-1] != 1]))

    # indexes = []

    # for i in range(len(imgs)):
    #     if labels[i][-1] != 1:
    #         indexes.append(i)
    
    # torch.save((imgs[indexes], labels[indexes]), "clean_dataset.pt")

    # images, labels_ = torch.load("dataset/clean_dataset.pt")
    # idx = random.randint(0, len(images) - 1)
    # x = imgs[idx]
    # transform = Resize((32, 32))

    # plt.imshow(x.numpy())
    # plt.show()
    # plt.imshow(transform(x.unsqueeze(0)).squeeze(0).numpy())
    # plt.show()
    loader = torch.utils.data.DataLoader(
        ds, shuffle=True, batch_size=32, drop_last=True, pin_memory=False, num_workers=0
    )

    for batch in loader:
        img, label = batch
        print(label[:, -1])
        break
