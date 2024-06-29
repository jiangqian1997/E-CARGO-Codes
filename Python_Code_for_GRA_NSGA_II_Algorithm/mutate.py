import numpy as np

def mutate(pops, pm, etaM, lb, rb):
    """
    Mutation operator using polynomial mutation.
    """
    nPop = pops.shape[0]
    for i in range(nPop):
        if np.random.rand() < pm:
            polyMutation(pops[i], etaM, lb, rb)
    return pops

def polyMutation(chr, etaM, lb, rb):
    """
    Polynomial mutation.
    """
    pos1, pos2 = np.sort(np.random.randint(0, len(chr), 2))
    pos2 += 1
    u = np.random.rand()
    if u < 0.5:
        delta = (2 * u) ** (1 / (etaM + 1)) - 1
    else:
        delta = 1 - (2 * (1 - u)) ** (1 / (etaM + 1))
    chr[pos1:pos2] += delta
    chr[chr < lb] = lb
    chr[chr > rb] = rb

def mutate_GRA(pops, pm, agent_number):
    """
    Mutation with Grey Relational Analysis.
    """
    nPop = pops.shape[0]
    for i in range(nPop):
        if np.random.rand() < pm:
            mutation(pops[i], agent_number)
    pops = np.unique(pops, axis=0)
    return pops

def mutation(chr, agent_number):
    """
    Mutation function that introduces variations.
    """
    agent_idx_list = np.arange(agent_number)
    pos1, pos2 = np.sort(np.random.randint(0, len(chr), 2))
    pos2 += 1
    new_chr_diff = np.setdiff1d(agent_idx_list, chr).tolist()
    if new_chr_diff:
        u = np.random.rand()
        if u < 0.5:  # Single-point mutation
            exchange_idx = np.random.randint(0, len(new_chr_diff))
            exchange_value = new_chr_diff[exchange_idx]
            chr[pos1] = exchange_value
        else:  # Multi-point mutation
            for idx in range(pos1, pos2):
                current_value = chr[idx]
                exchange_idx = np.random.randint(0, len(new_chr_diff))
                exchange_value = new_chr_diff.pop(exchange_idx)
                chr[idx] = exchange_value
                new_chr_diff.append(current_value)