"""
Fitness of populations or individuals
"""
import numpy as np
from function import *

def fitness_TAP(Q, Qp, pops, func):
    # Calculate fitness of populations or individuals
    # If 1D, reshape to 2D
    if pops.ndim == 1:
        pops = pops.reshape(1, len(pops))
    nPop = pops.shape[0]
    fits = np.array([func(Q, Qp, pops[i]) for i in range(nPop)])
    return fits

def transform_Q(L, Q):
    Transformed_Q_mat = np.zeros((len(Q), sum(L)))
    current_start_idx = 0
    accumulated_L_list = np.cumsum(L)
    for j in range(len(accumulated_L_list)):
        for i in range(len(Q)):
            Transformed_Q_mat[i][current_start_idx:accumulated_L_list[j]] = Q[i][j]
        current_start_idx = accumulated_L_list[j]
    return Transformed_Q_mat