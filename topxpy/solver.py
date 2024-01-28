import numpy as np
import scipy as sp

from numpy import kron, ones, zeros, sqrt, ceil, eye, sum, mean
from numpy.random import rand

from scipy import sparse
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, linalg

reshape = lambda x, sh: np.reshape(x, sh, order='F')

from numpy.matlib import repmat
import matplotlib.pyplot as plt

from ipypb import ipb

from dataclasses import dataclass
from typing import Union

import topxpy as top


class Solver:
    @dataclass
    class Physics:
        volfrac: float
        nu: float
        E0: float = 1.0
        Emin: float = 1e-9

        @property
        def Q_magnitude(self):
            return 1.0/(1 - self.nu**2)

    def __init__(
        self,
        nelx=64,
        nely=64,
        volfrac=0.4,
        penal=5.5,
        opt_step=0.5,
        opt_tol=1e-3,
        filterType=1,
        E0=1.0,
        Emin=1e-9,
        nu=0.3,
        rmin=1.3,
        maxit=100,
    ):
        self.physics = self.Physics(volfrac=volfrac, nu=nu, E0=E0, Emin=Emin)
        self.fem = top.fem.FEM(self, nelx=nelx, nely=nely)
        self.opt = top.opt.Optimizer(
            solver=self,
            penal=penal,
            tol=opt_tol,
            step=opt_step,
            filterType=filterType,
            rmin=rmin,
            maxit=maxit,
        )
        self.penal = penal
        self.Q = None

    


    def compute_Q_grad(self, xPhys, U):
        xPhys = xPhys.flatten()

        penal = self.penal
        edofMat = self.fem.edofMat
        nelx = self.fem.nelx
        nely = self.fem.nely
        KE = self.fem.KE
        E0 = self.physics.E0
        Emin = self.physics.Emin

        qe = [[None] * 3 for j in range(3)]
        Q = zeros([3, 3], dtype=float)
        dQ = [[None] * 3 for j in range(3)]
        for i in range(3):
            for j in range(3):
                U1 = U[:, i]
                U2 = U[:, j]
                qe[i][j] = sum((U1[edofMat].dot(KE)) * U2[edofMat], axis=1) / (
                    nelx * nely
                )
                Q[i, j] = sum((Emin + xPhys**penal * (E0 - Emin)) * qe[i][j])
                dQ[i][j] = penal * (E0 - Emin) * xPhys ** (penal - 1) * qe[i][j]
        return Q, dQ

    def compute_loss_grad(self, Q, dQ, loop, lossType=1):
        if lossType == 1:  # bulk modulus
            c = -(Q[0, 0] + Q[1, 1] + Q[0, 1] + Q[1, 0])
            dc = -(dQ[0][0] + dQ[1][1] + dQ[0][1] + dQ[1][0])
        elif lossType == 2:  # shear modulus
            c = -Q[2, 2]
            dc = -dQ[2][2]
        else:  # negative Poisson ratio
            delta = 0.5
            beta = 0.9
            # c = Q[0, 1] + Q[1, 0] - delta*Q[2, 2]
            c = Q[0, 1] / Q[0, 0]
            # c = -Q(3,3) + beta ** loop*(Q(1,2) + Q(2,1) );
            dc = -dQ[2, 2] + (dQ[0][1] + dQ[1][0]) * beta**loop
        return c, dc

    def update_design(self, x, dc, dv):
        # UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
        volfrac = self.physics.volfrac
        opt_step = self.opt.step
        x = x.flatten()
        l1 = 0
        l2 = 1e9
        while l2 - l1 > 1e-9:  # зачем оно вообще тут?
            lmid = 0.5 * (l2 + l1)
            val = -dc / dv / lmid
            val = abs(
                val
            )  # this was added solely as a dirty hack s.t. neg poisson loss does not crash with complex numbers
            xnew = np.maximum(
                0,
                np.maximum(
                    x - opt_step,
                    np.minimum(1, np.minimum(x + opt_step, x * sqrt(val))),
                ),
            )
            if self.opt.filter.type == 1:
                xPhys = xnew
            elif self.opt.filter.type == 2:
                sh = xPhys.shape
                xPhys = self.opt.filter.apply(xnew.flatten())
                xPhys = xPhys.reshape(sh)

            if mean(xPhys) > volfrac:
                l1 = lmid
            else:
                l2 = lmid
        return xnew, xPhys

    def optimize(
        self,
        max_opt_it=None,
        x0=None,
        drawingFrequency=5,
        linsolve_method='spsolve',
        linsolve_maxiter=100,
        verbose=False,
        draw=False,
        rA=0.0,
    ):
        '''returns: True if converged to a reasonable material'''
        opt_tol = self.opt.tol
        nelx, nely = self.fem.nelx, self.fem.nely
        volfrac = self.physics.volfrac
        if max_opt_it is None:
            max_opt_it = self.opt.maxit

        # # INITIALIZE ITERATION
        change = 1
        loop = 0
        if x0 is None:
            self.x = top.geom.cell_with_circle_hole(
                nelx=nelx, nely=nely, volfrac=volfrac, rA=rA
            )
        else:
            self.x = x0

        self.x = (
            volfrac - self.x * volfrac / 2
        )  # normalization hack, otherwise SIMP will not converge. So the x0 mask must be in range [volfrac/2, volfrac]
        self.x = self.x.flatten()
        self.xPhys = self.x

        while (loop < max_opt_it) and (change > opt_tol):
            U, status = self.fem.solve_fem(
                self.xPhys, method=linsolve_method, linsolve_maxiter=linsolve_maxiter
            )
            self.Q, dQ = self.compute_Q_grad(self.xPhys, U)
            c, dc = self.compute_loss_grad(self.Q, dQ, loop=loop)
            dc, dv = self.opt.filter.modify_sensitivities(dc=dc, x=self.x)
            self.xnew, self.xPhys = self.update_design(self.x, dc, dv)
            change = np.max(abs(self.xnew - self.x))
            self.x = self.xnew

            if verbose:
                print(loop, c, mean(self.xPhys), change)
            if draw:
                if loop % drawingFrequency == 0:
                    plt.figure()
                    plt.imshow(repmat(self.xPhys.reshape(nely, nelx), 2, 2))
                    plt.clim(0, 1)
            loop += 1
            if loop > 10 and self.struct_failed(self.xPhys):
                print('Converged to a broken material cell')
                return False
            elif loop > 50 and top.crit.is_gray(self.xPhys):
                print('Converged to a gray material cell')
                return False

        if top.crit.is_gray(self.xPhys):
            print('Converged to a gray material cell')
            return False
        else:
            return True

    def compute_Q(self, img):
        U, _ = self.fem.solve_fem(img.flatten())
        Q, _ = self.compute_Q_grad(img.flatten(), U)
        return Q

    def struct_failed(self, img):
        Q = self.compute_Q(img)
        failed = top.crit.first_criterion_of_failure(
            Q
        ) or top.crit.second_criterion_of_failure(Q)
        return failed

    def properties(self, img):
        Q = self.compute_Q(img)
        data = top.crit.Q_to_dict(Q)
        data['is_gray'] = int(top.crit.is_gray(img))
        data['volfrac'] = np.mean(img)
        data['is_broken'] = int(
            top.crit.first_criterion_of_failure(Q)
            or top.crit.second_criterion_of_failure(Q)
        )
        return data
