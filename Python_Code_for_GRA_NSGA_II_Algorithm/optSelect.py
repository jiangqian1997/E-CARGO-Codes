"""
Population Merging and Selection
"""
import numpy as np
from nonDominationSort import *

def optSelect(pops, fits, chrPops, chrFits):
    """Population merging and optimization
    Return:
        newPops, newFits
    """
    nPop, nChr = pops.shape
    nF = fits.shape[1]
    newPops = np.zeros((nPop, nChr))
    newFits = np.zeros((nPop, nF))
    # Merge parent and offspring populations to form a new population
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergeFits = np.concatenate((fits, chrFits), axis=0)
    MergeRanks = nonDominationSort(MergePops, MergeFits)
    MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks)

    indices = np.arange(MergePops.shape[0])
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]  # Indices of current rank r
    while i + len(rIndices) <= nPop:
        newPops[i:i+len(rIndices)] = MergePops[rIndices]
        newFits[i:i+len(rIndices)] = MergeFits[rIndices]
        r += 1  # Increase current rank
        i += len(rIndices)
        rIndices = indices[MergeRanks == r]  # Indices of current rank r
    if i < nPop:
        rDistances = MergeDistances[rIndices]  # Crowding distances of current rank
        rSortedIdx = np.argsort(rDistances)[::-1]  # Sort by distance in descending order
        surIndices = rIndices[rSortedIdx[:(nPop-i)]]
        newPops[i:] = MergePops[surIndices]
        newFits[i:] = MergeFits[surIndices]
    return (newPops, newFits)

def optSelect_GRA(pops, fits, chrPops, Q, Qp, func):
    """Population merging and optimization with GRA
    Return:
        newPops, newFits
    """
    nPop, nChr = pops.shape
    nF = fits.shape[1]
    newPops = np.zeros((nPop, nChr))
    newFits = np.zeros((nPop, nF))
    # Merge parent and offspring populations to form a new population
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergePops = np.unique(MergePops, axis=0)
    MergeFits = fitness_TAP(Q, Qp, MergePops, func)
    MergeRanks = nonDominationSort(MergePops, MergeFits)
    MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks)

    indices = np.arange(MergePops.shape[0])
    r = 0
    I = 0
    rIndices = indices[MergeRanks == r]  # Indices of current rank r
    while I + len(rIndices) <= nPop:
        newPops[I:I+len(rIndices)] = MergePops[rIndices]
        newFits[I:I+len(rIndices)] = MergeFits[rIndices]
        r += 1  # Increase current rank
        I += len(rIndices)
        rIndices = indices[MergeRanks == r]  # Indices of current rank r
    if I < nPop:
        rDistances = MergeDistances[rIndices]  # Crowding distances of current rank
        rSortedIdx = np.argsort(rDistances)[::-1]  # Sort by distance in descending order
        surIndices = rIndices[rSortedIdx[:(nPop-I)]]
        newPops[I:] = MergePops[surIndices]
        newFits[I:] = MergeFits[surIndices]
    return (newPops, newFits)