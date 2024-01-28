import numpy as np
from numpy import kron, ones, zeros, sqrt, ceil, eye, sum, mean
from numpy.random import rand

from scipy import sparse
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, linalg

reshape = lambda x, sh: np.reshape(x, sh, order='F')

from numpy.matlib import repmat
from topxpy.linalg import stack2d, stack2d_sparse, solve_linear_sparse


class FEM:
    def __init__(self, solver, nelx=64, nely=64):
        self.solver = solver
        self.nely = nely
        self.nelx = nelx
        self.edofMat = None
        self.KE = None
        self.iK = None
        self.jK = None
        self.d1 = None
        self.d2 = None
        self.d3 = None
        self.d4 = None
        self.ufixed = None
        self.wfixed = None

    def create_fem(self):
        nelx = self.nelx
        nely = self.nely
        nu = self.solver.physics.nu

        A11 = np.array(
            [[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]]
        )
        A12 = np.array(
            [[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]]
        )
        B11 = np.array(
            [[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]]
        )
        B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])

        KE = (
            1
            / (1 - nu**2)
            / 24
            * (
                stack2d([[A11, A12], [A12.T, A11]])
                + nu * stack2d([[B11, B12], [B12.T, B11]])
            )
        )

        nodenrs = reshape(
            np.arange(1, (1 + nelx) * (1 + nely) + 1), [1 + nely, 1 + nelx]
        )

        edofVec = reshape(2 * nodenrs[0:-1, 0:-1] + 1, [nelx * nely, 1])
        edofMat = (
            repmat(edofVec, 1, 8)
            + repmat(
                np.hstack([[0, 1], 2 * nely + np.array([2, 3, 0, 1]), [-2, -1]]),
                nelx * nely,
                1,
            )
            - 1
        )

        iK = reshape(kron(edofMat, ones([8, 1], dtype=int)).T, [64 * nelx * nely, 1])
        jK = reshape(kron(edofMat, ones([1, 8], dtype=int)).T, [64 * nelx * nely, 1])

        # PERIODIC BOUNDARY CONDITIONS
        e0 = eye(3)

        alldofs = np.arange(0, 2 * (nely + 1) * (nelx + 1))

        n1 = np.hstack([nodenrs[-1, [0, -1]], nodenrs[0, [-1, 0]]])
        d1 = reshape(np.array([2 * n1 - 1, 2 * n1]), [1, 8])
        n3 = np.hstack([nodenrs[1:-1, 0].T, nodenrs[-1, 1:-1]]).reshape(1, -1)
        d3 = reshape(np.vstack([(2 * n3 - 1), 2 * n3]), [1, 2 * (nelx + nely - 2)])
        n4 = np.hstack([nodenrs[1:-1, -1].T, nodenrs[0, 1:-1]]).reshape([1, -1])
        d4 = reshape(np.vstack([(2 * n4 - 1), 2 * n4]), [1, 2 * (nelx + nely - 2)])

        d1 -= 1
        d3 -= 1
        d4 -= 1

        d2 = set(alldofs) - set(d1.squeeze()).union(set(d3.squeeze())).union(
            set(d4.squeeze())
        )
        d2 = np.array(sorted(list(d2))).reshape(1, -1)

        ufixed = zeros([8, 3])
        for j in range(3):
            M = np.array([[e0[0, j], e0[2, j] / 2], [e0[2, j] / 2, e0[1, j]]])
            ufixed[2:4, j] = M.dot(np.array([[nelx], [0]])).squeeze()
            ufixed[6:8, j] = M.dot(np.array([[0], [nely]])).squeeze()
            ufixed[4:6, j] = ufixed[2:4, j] + ufixed[6:8, j]

        wfixed = np.vstack(
            [repmat(ufixed[2:4, :], nely - 1, 1), repmat(ufixed[6:8, :], nelx - 1, 1)]
        )

        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.edofMat = edofMat
        self.ufixed = ufixed
        self.wfixed = wfixed
        self.iK = iK
        self.jK = jK
        self.KE = KE

    def solve_fem(self, xPhys, U0=None, method='spsolve', linsolve_maxiter=100):
        Emin = self.solver.physics.Emin
        E0 = self.solver.physics.E0
        penal = self.solver.opt.penal
        if self.KE is None:
            self.create_fem()

        nelx = self.nelx
        nely = self.nely
        KE = self.KE
        iK = self.iK
        jK = self.jK
        d1 = self.d1
        d2 = self.d2
        d3 = self.d3
        d4 = self.d4
        ufixed = self.ufixed
        wfixed = self.wfixed

        if U0 is None:
            U = zeros([2 * (nely + 1) * (nelx + 1), 3])
        else:
            U = U0

        # FE-ANALYSIS
        v1 = KE.reshape(-1, 1)
        v2 = xPhys.reshape(1, -1)
        sK = reshape(v1 * (Emin + v2**penal * (E0 - Emin)), [64 * nelx * nely, 1])
        K = csr_matrix((sK.squeeze(), (iK.squeeze(), jK.squeeze())))
        K = (K + K.T) / 2
        A = K[d2.T, d2]
        B1 = K[d2.T, d3]
        B2 = K[d2.T, d4]
        B = B1 + B2
        C = K[d3.T, d2] + K[d4.T, d2]
        D = K[d3.T, d3] + K[d4.T, d3] + K[d3.T, d4] + K[d4.T, d4]
        Kr = stack2d_sparse([[A, B], [C, D]])
        # Kr = csr_matrix(Kr) # !!! FIX_ME. Conversion between sparse types may be slow

        U[d1, :] = ufixed
        rhs_c1 = -stack2d_sparse([[K[d2.T, d1]], [K[d3.T, d1] + K[d4.T, d1]]])
        rhs_c2 = -stack2d_sparse([[K[d2.T, d4]], [K[d3.T, d4] + K[d4.T, d4]]])
        rhs = rhs_c1.dot(ufixed) + rhs_c2.dot(wfixed)
        self.Kr = Kr
        self.rhs = rhs

        U[np.hstack([d2, d3]), :], status = solve_linear_sparse(
            Kr, rhs, method=method, linsolve_maxiter=linsolve_maxiter
        )  # linalg.spsolve(Kr, rhs)
        U[d4.flatten(), :] = U[d3.flatten(), :] + wfixed
        return U, status
