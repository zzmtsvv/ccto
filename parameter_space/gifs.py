from tqdm import tqdm as ptsd
from typing import List
import torch
from generator import Generator
from weight_deformator import ConstantWeightDeformator
from videos import make_video


def generate_videos(eigenvectors: List[torch.Tensor]):


    zs = torch.randn((4, 128))
 
    print('Making videos...')
    for i, eigenvector in ptsd(enumerate(eigenvectors)):
        # eigenvector = eigenvector.cuda()

        for amplitude in [10, 50]:# 100, 200, 500]:
            generator = Generator(1, num_blocks=[2, 2, 2, 2])
            generator.load_state_dict(torch.load("aae_unconditional_generator.pt", map_location="cpu"))

            wd = ConstantWeightDeformator(
                generator=generator,
                indexes=[2, 1, 2],
                direction=eigenvector
            )

            clip_path = f"directions/direction{i}/amplitude{amplitude}.gif"

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
    eigenvectors = torch.load("eigenvectors_alexnet.pt", map_location="cpu")

    generate_videos(eigenvectors)
