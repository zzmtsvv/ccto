from numpy.matlib import repmat
from numpy import sqrt
import numpy as np
import porespy as ps

# this is initial guess geometry shaping


def create_circle_matlab_style(nelx, nely, volfrac, R_frac=5.0):
    # x = repmat(volfrac, nely, nelx)
    x = repmat(1.0, nely, nelx)
    for i in range(1, nelx + 1):
        for j in range(1, nely + 1):
            if (
                sqrt((i - nelx / 2 - 0.5) ** 2 + (j - nely / 2 - 0.5) ** 2)
                < min(nelx, nely) / R_frac
            ):
                # x[j, i] = volfrac * 0.5
                x[j, i] = 0
    return x


def cell_with_circle_hole(nelx, nely, volfrac, rA=0, R_frac=5.0):
    mask = create_circle_matlab_style(nelx, nely, volfrac, R_frac=R_frac)
    # x = volfrac * np.ones([nely, nelx]) - volfrac/2*mask
    return mask - rA * np.random.rand(nely, nelx)


class Geometry:
    def __init__(self, nelx, nely, fun):
        self.nelx = nelx
        self.nely = nely
        self.fun = fun

    def create_meshgrid(self):
        nelx, nely = self.nelx, self.nely
        XX, YY = np.meshgrid(np.linspace(-1, 1, nelx), np.linspace(-1, 1, nely))
        return XX, YY

    def create_mask(self):
        XX, YY = self.create_meshgrid()
        return self.fun(XX, YY)


def sigmoid(x, sharpness=300):
    z = 1.0 / (1.0 + np.exp(-sharpness * x))
    return z


def circ_fun(x, y, R=0.5, sharpness=300):
    z = (x) ** 2 + (y) ** 2 - R**2
    return sigmoid(z, sharpness)


def ellipse_fun(x, y, R=0.5, sharpness=300):
    z = (x) ** 2 + (y) ** 2 - R**2
    return sigmoid(z, sharpness)


def half_space_fun(x, y, L=0.0, angle=0.0, sharpness=300):
    z = x * np.sin(angle) + y * np.cos(angle) + L
    return sigmoid(z, sharpness)


def stripe(x, y, L=0.25, angle=0, sharpness=300):
    f1 = lambda x, y: half_space_fun(x, y, L=L, angle=angle, sharpness=sharpness)
    f2 = lambda x, y: half_space_fun(
        x, y, L=L, angle=angle + np.pi, sharpness=sharpness
    )
    return f1(x, y) * f2(x, y)


def hstripe(x, y, L=0.25, sharpness=250):
    return stripe(x, y, L=L, angle=0, sharpness=sharpness)


def vstripe(x, y, L=0.25, sharpness=250):
    return stripe(x, y, L=L, angle=np.pi / 2, sharpness=sharpness)


def cross_plus(x, y, L=0.25, sharpness=250):
    f1 = lambda x, y: hstripe(x, y, L=L, sharpness=sharpness)
    f2 = lambda x, y: vstripe(x, y, L=L, sharpness=sharpness)
    return np.clip(f1(x, y) + f2(x, y), 0, 1)


def diagstripe1(x, y, L=0.25, sharpness=250):
    return stripe(x, y, L=L, angle=np.pi / 4, sharpness=sharpness)


def diagstripe2(x, y, L=0.25, sharpness=250):
    return stripe(x, y, L=L, angle=-np.pi / 4, sharpness=sharpness)


def cross_x(x, y, L=0.25, sharpness=250):
    f1 = lambda x, y: diagstripe1(x, y, L=L, sharpness=sharpness)
    f2 = lambda x, y: diagstripe2(x, y, L=L, sharpness=sharpness)
    return np.clip(f1(x, y) + f2(x, y), 0, 1)


def random_geometry(algorithm=None, porosity=None, shape=(64, 64)):
    ALL_ALGORITHMS = ['blobs', 'polysph', 'sph', 'voronoi']
    if porosity is None:
        porosity = 0.5 + 0.4 * np.random.rand()

    if algorithm is None:
        algorithm = np.random.choice(ALL_ALGORITHMS)

    if algorithm == 'blobs':
        scale = np.random.rand() * 2.0
        blobiness = 10 ** (-scale)
        img = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=blobiness)
    elif algorithm == 'polysph':
        import scipy.stats as spst

        dist = spst.norm(loc=10, scale=5)
        img = ps.generators.polydisperse_spheres(
            shape=shape, porosity=porosity, dist=dist
        )
    elif algorithm == 'sph':

        r = 10.5
        img = ps.generators.overlapping_spheres(shape=shape, r=r, porosity=porosity)
    elif algorithm == 'voronoi':
        ncells = int(10 + 30 * np.random.rand())
        r = int(1 + 2 * np.random.rand())
        img = 1 - ps.generators.voronoi_edges(
            shape=shape, ncells=ncells, flat_faces=False, r=r
        )

    return img.astype(np.float32)


def add_noise(img, A=0.1):
    return img * (1.0 - A * np.random.rand(*img.shape))


def add_noise_n(img, A=0.1):
    img += A * np.random.randn(*img.shape)
    img = np.clip(img, 0, 1)
    return img


def compress_along_axis(img, tiling=(2,2)):
    step_x, step_y = tiling
    return np.tile(img[::step_x,::step_y], tiling)[:img.shape[0], :img.shape[1]]