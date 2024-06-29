"""
Selection Operator
"""
import random
import numpy as np

def select1(pool, pops, fits, ranks, distances):
    # One-to-one tournament selection
    # pool: Size of the newly generated population
    nPop, nChr = pops.shape
    nF = fits.shape[1]
    newPops = np.zeros((pool, nChr))
    newFits = np.zeros((pool, nF))
    indices = np.arange(nPop).tolist()
    i = 0
    while i < pool:
        idx1, idx2 = random.sample(indices, 2)  # Randomly select two individuals
        idx = compare(idx1, idx2, ranks, distances)
        newPops[i] = pops[idx]
        newFits[i] = fits[idx]
        i += 1
    return newPops, newFits

def compare(idx1, idx2, ranks, distances):
    # Return the better index
    if ranks[idx1] < ranks[idx2]:
        idx = idx1
    elif ranks[idx1] > ranks[idx2]:
        idx = idx2
    else:
        if distances[idx1] <= distances[idx2]:
            idx = idx2
        else:
            idx = idx1
    return idx