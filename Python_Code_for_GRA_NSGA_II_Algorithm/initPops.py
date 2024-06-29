from random import random
import numpy as np
import random

def initPops_TAP(npop, nchr, agent_number, initial_pops=[]):
    """
    Population Initialization
    """
    pops = initial_pops.tolist()
    if not pops:
        initial_pops = np.unique(initial_pops, axis=0)
    else:
        pops = []
    current_pop_number = npop - len(initial_pops)
    while current_pop_number > 0:
        current_pop = np.array(random.sample(range(agent_number), nchr)).tolist()
        if current_pop not in pops:
            pops.append(current_pop)
            current_pop_number -= 1
    pops = np.array(pops)
    return pops