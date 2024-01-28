import torch
from generator import Generator

state_dict = torch.load("/Users/alexzzmtsvv/Desktop/aae_tiling.pt", map_location="cpu")

print(state_dict.keys())


generator = Generator(in_channels=1, num_blocks=[2, 2, 2, 2])
generator.load_state_dict(state_dict["generator"])
# torch.save(state_dict["generator"], "aae_unconditional_generator.pt")


# from matplotlib import pyplot as plt
from torchvision.utils import make_grid


generated_images = generator.sample(64).cpu()
generated_images = make_grid(
        generated_images, nrow=1, normalize=True, value_range=(-1, 1)).numpy().transpose(1, 2, 0)
print(generated_images.shape)

# plt.imshow(generated_images, cmap="gray")
# plt.show()

# for module in generator.modules():
#     print(module)

