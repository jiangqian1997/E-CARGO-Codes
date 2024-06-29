"""
Fast Non-Dominated Sorting
"""
import random
import numpy as np
from fitness import fitness_TAP
from function import *

def nonDominationSort_min(pops, fits):
    """Fast Non-Dominated Sorting Algorithm
    Params:
        pops: Population, nPop * nChr array
        fits: Fitness, nPop * nF array
    Return:
        ranks: Rank of each individual, 1D array
    """
    nPop = pops.shape[0]
    nF = fits.shape[1]  # Number of objective functions
    ranks = np.zeros(nPop, dtype=np.int32)
    nPs = np.zeros(nPop)  # Number of solutions dominating each individual p
    sPs = []  # Set of solutions dominated by each individual
    for i in range(nPop):
        iSet = []  # Dominated set of solution i
        for j in range(nPop):
            if i == j:
                continue
            isDom1 = fits[i] <= fits[j]
            isDom2 = fits[i] < fits[j]
            # Check if i dominates j
            if sum(isDom1) == nF and sum(isDom2) >= 1:
                iSet.append(j)
            # Check if i is dominated by j
            if sum(~isDom2) == nF and sum(~isDom1) >= 1:
                nPs[i] += 1
        sPs.append(iSet)  # Add indices of solutions dominated by i
    r = 0  # Current rank, lower is better
    indices = np.arange(nPop)
    while sum(nPs == 0) != 0:
        rIdices = indices[nPs == 0]  # Indices with zero domination count
        ranks[rIdices] = r
        for rIdx in rIdices:
            iSet = sPs[rIdx]
            nPs[iSet] -= 1
        nPs[rIdices] = -1  # Set domination count of current rank to negative
        r += 1
    return ranks

def nonDominationSort(pops, fits):
    """Fast Non-Dominated Sorting Algorithm
    Params:
        pops: Population, nPop * nChr array
        fits: Fitness, nPop * nF array
    Return:
        ranks: Rank of each individual, 1D array
    """
    nPop = pops.shape[0]
    nF = fits.shape[1]  # Number of objective functions
    ranks = np.zeros(nPop, dtype=np.int32)
    nPs = np.zeros(nPop)  # Number of solutions dominating each individual p
    sPs = []  # Set of solutions dominated by each individual
    for i in range(nPop):
        iSet = []  # Dominated set of solution i
        for j in range(nPop):
            if i == j:
                continue
            isDom1 = fits[i] >= fits[j]
            isDom2 = fits[i] > fits[j]
            # Check if i dominates j
            if sum(isDom1) == nF and sum(isDom2) >= 1:
                iSet.append(j)
            # Check if i is dominated by j
            if sum(~isDom2) == nF and sum(~isDom1) >= 1:
                nPs[i] += 1
        sPs.append(iSet)  # Add indices of solutions dominated by i
    r = 0  # Current rank, lower is better
    indices = np.arange(nPop)
    while sum(nPs == 0) != 0:
        rIdices = indices[nPs == 0]  # Indices with zero domination count
        ranks[rIdices] = r
        for rIdx in rIdices:
            iSet = sPs[rIdx]
            nPs[iSet] -= 1
        nPs[rIdices] = -1  # Set domination count of current rank to negative
        r += 1
    return ranks

# Crowding Distance Sorting Algorithm
def crowdingDistanceSort(pops, fits, ranks):
    """Crowding Distance Sorting Algorithm
    Params:
        pops: Population, nPop * nChr array
        fits: Fitness, nPop * nF array
        ranks: Rank of each individual, 1D array
    Return:
        dis: Crowding distance of each individual, 1D array
    """
    nPop = pops.shape[0]
    nF = fits.shape[1]  # Number of objectives
    dis = np.zeros(nPop)
    nR = ranks.max()  # Maximum rank
    indices = np.arange(nPop)
    for r in range(nR + 1):
        rIdices = indices[ranks == r]  # Indices of current rank population
        rPops = pops[ranks == r]  # Current rank population
        rFits = fits[ranks == r]  # Fitness of current rank population
        rSortIdices = np.argsort(rFits, axis=0)  # Indices sorted by fitness
        rSortFits = np.sort(rFits, axis=0)
        fMax = np.max(rFits, axis=0)
        fMin = np.min(rFits, axis=0)
        n = len(rIdices)
        for i in range(nF):
            orIdices = rIdices[rSortIdices[:, i]]  # Original positions
            j = 1
            while n > 2 and j < n - 1:
                if fMax[i] != fMin[i]:
                    dis[orIdices[j]] += (rSortFits[j + 1, i] - rSortFits[j - 1, i]) / \
                        (fMax[i] - fMin[i])
                else:
                    dis[orIdices[j]] = np.inf
                j += 1
            dis[orIdices[0]] = np.inf
            dis[orIdices[n - 1]] = np.inf
    return dis
