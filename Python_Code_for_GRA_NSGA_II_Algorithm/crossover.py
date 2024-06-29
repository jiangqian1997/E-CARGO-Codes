"""
Crossover Operator using Simulated Binary Crossover (SBX)
"""
import numpy as np

def SBX(chr1, chr2, etaC, lb, rb):
    # Simulated Binary Crossover
    pos1, pos2 = np.sort(np.random.randint(0, len(chr1), 2))
    pos2 += 1  # [pos1:pos2] right boundary is open, needs +1
    u = np.random.rand()
    if u <= 0.5:
        gamma = (2 * u) ** (1 / (etaC + 1))
    else:
        gamma = (1 / (2 * (1 - u))) ** (1 / (etaC + 1))
    x1 = chr1[pos1:pos2]
    x2 = chr2[pos1:pos2]
    chr1[pos1:pos2], chr2[pos1:pos2] = 0.5 * ((1 + gamma) * x1 + (1 - gamma) * x2), \
                                        0.5 * ((1 - gamma) * x1 + (1 + gamma) * x2)
    # Check for constraints
    chr1[chr1 < lb] = lb
    chr1[chr1 > rb] = rb
    chr2[chr2 < lb] = lb
    chr2[chr2 > rb] = rb

def crossover_GRA(pops, pc, agent_number):
    # Copy parent population to prevent structure changes
    chrPops = pops.copy()
    nPop = chrPops.shape[0]
    for i in range(0, nPop, 2):
        if np.random.rand() < pc:
            exchangeProcess(chrPops[i], chrPops[i + 1], agent_number)  # Crossover operation
    # Remove duplicate populations
    chrPops = np.unique(chrPops, axis=0)
    return chrPops

def exchangeProcess(chr1, chr2, agent_number):
    agent_idx_list = np.arange(agent_number)
    pos1, pos2 = np.sort(np.random.randint(0, len(chr1), 2))
    pos2 += 1  # [pos1:pos2] right boundary is open, needs +1
    x1 = chr1[pos1:pos2].copy()
    x2 = chr2[pos1:pos2].copy()
    chr1[pos1:pos2], chr2[pos1:pos2] = x2, x1
    # Check for multiple roles assigned to one agent and correct
    new_chr1_diff = np.setdiff1d(agent_idx_list, chr1).tolist()
    new_chr2_diff = np.setdiff1d(agent_idx_list, chr2).tolist()
    for idx in range(pos1, pos2):
        if np.sum(chr1 == chr1[idx]) > 1:
            exchange_idx = np.random.randint(0, len(new_chr1_diff))
            exchange_value = new_chr1_diff.pop(exchange_idx)
            chr1[idx] = exchange_value
        if np.sum(chr2 == chr2[idx]) > 1:
            exchange_idx = np.random.randint(0, len(new_chr2_diff))
            exchange_value = new_chr2_diff.pop(exchange_idx)
            chr2[idx] = exchange_value