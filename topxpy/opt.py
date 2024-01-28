import numpy as np
import scipy as sp

from numpy import kron, ones, zeros, sqrt, ceil, eye, sum, mean
from numpy.random import rand

from scipy import sparse
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, linalg
from dataclasses import dataclass
from typing import Union


class Optimizer:
    class Filter:
        def __init__(self, type=1, rmin=1.3, nelx=64, nely=64):
            self.type = type
            self.rmin = rmin
            self.H, self.Hs = self.prepare(nelx=nelx, nely=nely, rmin=rmin)

        def prepare(self, nelx, nely, rmin=None):
            CR = int(ceil(rmin)) - 1
            iH = zeros([nelx * nely * (2 * CR + 1) ** 2, 1], dtype=int)
            jH = zeros(iH.shape, dtype=int)
            sH = zeros(iH.shape, dtype=float)
            k = -1
            for i1 in range(1, nelx + 1):
                for j1 in range(1, nely + 1):
                    e1 = (i1 - 1) * nely + j1 - 1
                    for i2 in range(max(i1 - CR, 1), min(i1 + CR, nelx) + 1):
                        for j2 in range(max(j1 - CR, 1), min(j1 + CR, nely) + 1):
                            e2 = (i2 - 1) * nely + j2 - 1
                            k = k + 1
                            iH[k] = e1
                            jH[k] = e2
                            sH[k] = max(0, rmin - sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2))

            H = csc_matrix((sH.flatten(), (iH.flatten(), jH.flatten())))
            Hs = np.array(H.sum(1)).flatten()
            return H, Hs

        def apply(self, v):
            H, Hs = self.H, self.Hs
            return H.dot(v.flatten()) / Hs

        def modify_sensitivities(self, x, dc):
            # FILTERING/MODIFICATION OF SENSITIVITIES

            dv = np.ones_like(x, dtype=float).flatten()

            if self.type == 1:
                dc = self.apply(x.flatten() * dc.flatten()) / np.maximum(
                    1e-3, x.flatten()
                )
            elif self.type == 2:
                dc = self.apply(dc.flatten())
                dv = self.apply(dv.flatten())
            return dc, dv

    def __init__(self, solver, tol, step, filterType=1, penal=1.3, rmin=1.3, maxit=100):
        self.solver = solver
        self.tol = tol
        self.step = step
        self.penal = penal
        self.filter = self.Filter(
            type=filterType,
            rmin=rmin,
            nelx=self.solver.fem.nelx,
            nely=self.solver.fem.nely,
        )
        self.maxit = maxit
