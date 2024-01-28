import numpy as np
import torch.nn.functional as F
from moviepy.editor import ImageSequenceClip
from matplotlib import pyplot as plt
import os
import torch
# import cv2
from torchvision.utils import make_grid

from generator import Generator
from weight_deformator import ConstantWeightDeformator


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
    
    imgs_total = []
    for shift in shifts:
        wd.deformate(*wd_deformate_arguments_builder(shift))
        with torch.no_grad():
            if interpolate is not None:
                imgs = F.interpolate(generator(z), size=interpolate)
                imgs = (imgs.cpu().numpy().transpose((0, 2, 3, 1)) + 1) / 2
            else:
                imgs = (generator(z).cpu().numpy().transpose((0, 2, 3, 1)) + 1) / 2
                # grid = make_grid(imgs, nrow=2, normalize=True, value_range=(-1, 1)).numpy().transpose(1, 2, 0)

                # plt.imsave(f"{clip_path}/{shift}.jpeg", grid)
            imgs = np.concatenate(imgs, axis=1)
        imgs_total.append(imgs)

    wd.disable_deformation()

    imgs_total = np.stack(imgs_total, axis=0)
    imgs_total = (imgs_total * 255).astype('uint8')

    clip = ImageSequenceClip(list(imgs_total), fps=30)
    clip.write_gif(clip_path, fps=30)

    # imgs_total = imgs_total[..., ::-1] # RGB -> BGR
    
    # out = cv2.VideoWriter(clip_path,
    #                       cv2.VideoWriter_fourcc('M','J','P','G'),
    #                       len(imgs_total) / gif_seconds,
    #                       (imgs_total.shape[2], imgs_total.shape[1]))
        
    # for img in imgs_total:
    #     out.write(img)
    # for img in imgs_total[::-1]:
    #     out.write(img)

    # out.release()

    # avi_clip = moviepy.VideoFileClip(clip_path)
    # avi_clip.write_videofile(clip_path_mp4, logger=None)
    # avi_clip.close()
    # os.remove(clip_path)