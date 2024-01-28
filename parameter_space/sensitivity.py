import numpy as np
import torch
from generator import Generator
from weight_deformator import ConstantWeightDeformator
from matplotlib import pyplot as plt
from torchvision.utils import make_grid


if __name__ == "__main__":
    step = 10 / 50.0
    shifts = np.arange(-10, 10 + step, step)
    shifts = shifts[4:80]
    print(shifts.shape)
    
    exit()
    eigenvector = torch.load("material_eigenvector.pt", map_location="cpu")
    model = Generator(1, num_blocks=[2, 2, 2, 2])
    model.load_state_dict(torch.load("aae_unconditional_generator.pt", map_location="cpu"))
    model.eval()

    noise = torch.load("z.pt", map_location="cpu")

    generations = model(noise)
    generations = make_grid(generations, nrow=8, normalize=True, value_range=(-1, 1)).numpy().transpose(1, 2, 0)
    plt.imshow(generations)
    plt.show()
    
    wd = ConstantWeightDeformator(model, eigenvector)
    wd.deformate(-10.0)

    generations = model(noise)
    generations = make_grid(generations, nrow=8, normalize=True, value_range=(-1, 1)).numpy().transpose(1, 2, 0)
    plt.imshow(generations)
    plt.show()

    # wd.disable_deformation()
