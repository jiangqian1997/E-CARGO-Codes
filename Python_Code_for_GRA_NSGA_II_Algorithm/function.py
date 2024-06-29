"""
Multi-objective optimization function, returns multiple objective values
"""
import numpy as np

def multiobjective_function(Q, Qp, T):
    y1 = 0
    y2 = 0
    if len(set(T)) != len(T):
        return y1, y2
    for idx in range(len(T)):
        y1 += Q[int(T[idx])][idx]
        y2 += Qp[int(T[idx])][idx]
    return y1, y2