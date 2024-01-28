import numpy as np
from scipy import linalg as la
import topxpy as top


Q_KEYS = ('Q_00', 'Q_01', 'Q_02', 'Q_11', 'Q_12', 'Q_22', 'is_gray', 'volfrac', 'is_broken')
IDX_Q_KEYS = dict()
for i, key in enumerate(Q_KEYS):
    IDX_Q_KEYS[key] = i


def dict_to_vec(data: dict):
    return np.array( [data[key] for key in Q_KEYS] )


def vec_to_dict(data):
    sample_dict = dict()
    for i, key in enumerate(Q_KEYS):
        sample_dict[key] = data[i]
    return sample_dict
    

def det2(Q):
    return np.linalg.det(Q[:2, :2])


def det3(Q):
    return np.linalg.det(Q)


def Q_to_vec(Q):
    return [Q[0, 0], Q[1, 1], Q[2, 2], Q[0, 1], Q[1, 2], Q[0, 2]]


def vec_to_Q(vec):
    Q = np.zeros([3, 3], dtype=float)
    Q[0, 0], Q[1, 1], Q[2, 2], Q[0, 1], Q[1, 2], Q[0, 2] = vec
    Q[1, 0] = Q[0, 1]
    Q[2, 1] = Q[1, 2]
    Q[2, 0] = Q[0, 2]
    return Q


def Q_to_dict(Q):
    d = {}
    for i in range(3):
        for j in range(3):
            if i <= j:
                d['Q_' + str(i) + str(j)] = Q[i, j]
    return d


def dict_to_Q(d):
    Q = np.zeros([3, 3], dtype=float)
    Q[0, 0] = d['Q_00']
    Q[1, 1] = d['Q_11']
    Q[2, 2] = d['Q_22']
    Q[0, 1] = d['Q_01']
    Q[1, 0] = d['Q_01']
    Q[0, 2] = d['Q_02']
    Q[2, 0] = d['Q_02']
    Q[1, 2] = d['Q_12']
    Q[2, 1] = d['Q_12']
    return Q


def first_criterion_of_failure(Q):
    FIRST_CRIT_CONST = 1e-7
    return det2(Q) < FIRST_CRIT_CONST


def second_criterion_of_failure(Q):
    SECOND_CRIT_CONST = 1000  # this constant means that there exist a direction in the material that is 1000 times stronger than another direction.
    _, s, _ = la.svd(Q)
    return s[0] / s[1] > SECOND_CRIT_CONST


def is_gray(x):
    THIRD_CRIT_CONST = 1e-1
    return np.var(x) / np.mean(x) < THIRD_CRIT_CONST

def compute_Q(img, sol):
    '''
    img is a 2D numpy array of floats in [0.0, 1.0]
    sol is a Solver instance
    '''
    U, _ = sol.fem.solve_fem(img.flatten())
    Q, _ = sol.compute_Q_grad(img.flatten(), U)
    return Q