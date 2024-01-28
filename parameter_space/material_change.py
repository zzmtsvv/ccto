import torch
import numpy as np
# from moviepy.editor import ImageSequenceClip
from torch.nn import functional as F
from torchvision.utils import make_grid
from typing import List
from tqdm import tqdm as ptsd
from weight_deformator import ConstantWeightDeformator
from generator import Generator
from matplotlib import pyplot as plt


def make_video(generator: Generator,
               z: torch.Tensor,
               wd: ConstantWeightDeformator,
               file_destination: str,
               shift_from,
               shift_to,
               step=2,
               gif_seconds=3.5,
               interpolate=None,
               wd_deformate_arguments_builder=lambda shift_: [shift_, ]):
    generator.eval()
    
    assert 'avi' not in file_destination and 'mp4' not in file_destination

    clip_path = file_destination

    shifts = np.arange(shift_from, shift_to + step, step)
    shifts = shifts[14:60]
    prev = None
    
    imgs_total = []
    i = 1
    for shift in shifts:
        wd.deformate(*wd_deformate_arguments_builder(shift))
        with torch.no_grad():
            if interpolate is not None:
                imgs = F.interpolate(generator(z), size=interpolate)
                imgs = (imgs.cpu().numpy().transpose((0, 2, 3, 1)) + 1) / 2
            else:
                imgs = generator(z).cpu()
                if prev is not None:
                    diff = (prev.sum(dim=(1, 2, 3)) - imgs.sum(dim=(1, 2, 3))).mean().item()
                    print(i + 16, diff)
                
                prev = imgs
                grid = make_grid(imgs, nrow=8, normalize=True, value_range=(-1, 1)).numpy().transpose(1, 2, 0)

                plt.imsave(f"{clip_path}/{i}.jpeg", grid)
                i += 1
            # imgs = np.concatenate(imgs, axis=1)
        imgs_total.append(grid)

    wd.disable_deformation()

    # imgs_total = np.stack(imgs_total, axis=0)
    # imgs_total = (imgs_total * 255).astype('uint8')

    # clip = ImageSequenceClip(list(imgs_total), fps=30)
    # clip.write_gif(clip_path, fps=30)



def meow(eigenvectors: List[torch.Tensor]):


    zs = torch.randn((64, 128))
    eigenvector = eigenvectors[4]
 
    print('Making videos...')


    for amplitude in [10]:# 100, 200, 500]:
        generator = Generator(1, num_blocks=[2, 2, 2, 2])
        generator.load_state_dict(torch.load("aae_unconditional_generator.pt", map_location="cpu"))

        wd = ConstantWeightDeformator(
            generator=generator,
            indexes=[2, 1, 2],
            direction=eigenvector
            )

        clip_path = f"material_change"
        make_video(
            generator=generator,
            z=zs,
            wd=wd,
            file_destination=clip_path,
            shift_from=-amplitude,
            shift_to=amplitude,
            step=amplitude / 50.,
            interpolate=None
            )


if __name__ == "__main__":
    # from index 16 until index 60 to avoid outliers
    # data = [
    #     0.0027795052155852318,
    #     0.0031186239793896675,
    #     0.0034452409017831087,
    #     0.003821911057457328,
    #     0.0043143536895513535,
    #     0.004798330366611481,
    #     0.005258710123598576,
    #     0.005706156138330698,
    #     0.006067884620279074,
    #     0.006577998399734497,
    #     0.0073325419798493385,
    #     0.008020629175007343,
    #     0.008558094501495361,
    #     0.009166195057332516,
    #     0.009785700589418411,
    #     0.010284649208188057,
    #     0.010863588191568851,
    #     0.011471301317214966,
    #     0.012006939388811588,
    #     0.012584779411554337,
    #     0.013611678034067154,
    #     0.013908102177083492,
    #     0.01446114107966423,
    #     0.015102416276931763,
    #     0.015738176181912422,
    #     0.015984660014510155,
    #     0.01661008968949318,
    #     0.017283881083130836,
    #     0.01773093454539776,
    #     0.018195150420069695,
    #     0.018892379477620125,
    #     0.019864898175001144,
    #     0.02056938037276268,
    #     0.021786142140626907,
    #     0.023368623107671738,
    #     0.027172857895493507,
    #     0.029913296923041344,
    #     0.03127506002783775,
    #     0.031731974333524704,
    #     0.03066862002015114,
    #     0.02901642397046089,
    #     0.029107125476002693,
    #     0.02903023362159729,
    #     0.02857252024114132,
    #     0.026980679482221603,
    # ]
    # print(np.mean(data))
    # print(np.std(data))
    # exit()
    
    eigenvectors = torch.load("eigenvectors_alexnet.pt", map_location="cpu")

    direction_eigenvector = eigenvectors[4]
    torch.save(direction_eigenvector, "direction_eigenvector.pt")
    # print(direction_eigenvector.shape)

    meow(eigenvectors)

    import imageio
    fps = 30
    filenames = [f"material_change/{i}.jpeg" for i in range(1, 46 + 1)]
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'2fpsmovie{fps}.gif', images, fps=fps)