import numpy as np
import matplotlib.pyplot as plt
import torch

# from edt import edt
import mcubes

# from edt import sdf as sdf2


def multiply(bin2d, M=4):
    tdat3 = torch.tensor(bin2d)
    multiplied = torch.hstack([torch.vstack([tdat3] * M)] * M)
    return multiplied


def convert2dto3d(img2d, Nz=3):
    return torch.stack([img2d] * Nz).permute([1, 2, 0])


def smooth_and_multiply(bin, band_radius=3.0, c_thick=1 / 2.5, x=4):
    dat = -mcubes.smooth_constrained(bin, band_radius=band_radius)
    dat = dat + band_radius * c_thick
    if x > 1:
        dat = multiply(dat, M=x)
    return dat


def make_STL_from_2d(dat, fname, M=10, relative_Z_size=1.0 / 3):
    LAST_LAYER_MARGIN = 0.0

    # multiply 2d pattern
    multiplied = multiply(dat, M=M)
    multiplied = convert2dto3d(multiplied, Nz=3)

    # adding the third dimension and doing marching cubes
    # A_new = max(multiplied.shape)*relative_Z_size/2.0

    m = torch.nn.ConstantPad3d(1, value=0)
    final = m(multiplied.unsqueeze(0)).squeeze()

    ver, faces = mcubes.marching_cubes(final.numpy(), 0.0)

    # centering X,Y and rescaling X,Y,Z
    As = np.max(ver, axis=0) - np.min(ver, axis=0)
    ver[:, 0] -= As[0] / 2
    ver[:, 1] -= As[1] / 2
    Aglob = max(As)
    ver /= Aglob

    print(np.max(ver, axis=0) - np.min(ver, axis=0))

    # rescaling the third dimension

    zets = ver[:, 2]
    Az = (max(zets) - min(zets)) / 2.0
    zets -= Az

    cur_indices = np.abs(zets) == Az
    zets[cur_indices] = zets[cur_indices] / Az * relative_Z_size / 2

    cur_indices = (np.abs(zets) < Az) * (np.abs(zets) > 0)
    zets[cur_indices] = zets[cur_indices] * (relative_Z_size / 2 - LAST_LAYER_MARGIN)

    ver[:, 2] = zets

    size_mm = 30.0

    ver *= size_mm

    # exporting the resulting meshgrid
    mcubes.export_obj(ver, faces, fname + '.obj')
    return ver, faces


def save_array_to_png(img, i, folder=''):
    fname = str(i)
    pic = img.astype(np.uint8)
    full_path = folder + fname + '.png'
    plt.imsave(full_path, pic, cmap='gray')
