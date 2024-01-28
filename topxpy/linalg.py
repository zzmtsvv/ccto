import numpy as np
import scipy as sp

from numpy import kron, ones, zeros, sqrt, ceil, eye, sum, mean
from numpy.random import rand

from scipy import sparse
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, linalg

METHODS = ['spsolve', 'gmres', 'bicg', 'bicgstab', 'cg', 'cgs', 'lgmres', 'gcrotmk']


def stack2d(dat):
    return np.vstack([np.hstack(row) for row in dat])


def stack2d_sparse(dat):
    return sparse.vstack([sparse.hstack(row) for row in dat])


def solve_linear_sparse(A, b, method='spsolve', tol=1e-3, linsolve_maxiter=100):
    if method == 'spsolve':
        x = linalg.spsolve(A, b)
        statuses = [0]
    else:
        xs = []
        statuses = []
        for b_col in b.T:
            if method == 'gmres':
                x, status = linalg.gmres(A, b_col, maxiter=linsolve_maxiter, tol=tol)
            elif method == 'bicg':
                x, status = linalg.bicg(A, b_col, maxiter=linsolve_maxiter, tol=tol)
            elif method == 'bicgstab':
                x, status = linalg.bicgstab(A, b_col, maxiter=linsolve_maxiter, tol=tol)
            elif method == 'cg':
                x, status = linalg.cg(A, b_col, maxiter=linsolve_maxiter, tol=tol)
            elif method == 'cgs':
                x, status = linalg.cgs(A, b_col, maxiter=linsolve_maxiter, tol=tol)
            elif method == 'lgmres':
                x, status = linalg.lgmres(
                    A, b_col, maxiter=linsolve_maxiter, tol=tol, atol=tol
                )
            elif method == 'gcrotmk':
                x, status = linalg.gcrotmk(
                    A, b_col, maxiter=linsolve_maxiter, tol=tol, atol=tol
                )
            else:
                raise NotImplementedError(
                    'only the following methods are implemented: ' + ', '.join(METHODS)
                )

            xs.append(x)
            statuses.append(x)
        x = np.stack(xs).squeeze().T

    return x, statuses
