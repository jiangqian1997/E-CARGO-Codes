"""

 * This program is to achieve the GRA-NSGA-II Algorithm in the paper named "Iterative Role Negotiation via the Bi-level GRA++ with Decision Tolerance".

 * Created by Qian Jiang Jan. 7, 2023

 * Please cite the paper as the following formation: Q. Jiang, D. Liu, H. Zhu, S. Wu, N. Wu, X. Luo, and Y. Qiao, "Iterative Role Negotiation via the Bi-level GRA++ with Decision Tolerance," IEEE Trans. Comput. Soc. Syst., early access, June 26, 2024, doi: 10.1109/TCSS.2024.3409893.

"""

import numpy as np
from initPops import *
from optSelect import optSelect
from nonDominationSort import *
from select1 import *
from fitness import *
from crossover import *
from mutate import *



def NSGA_II_GRA(nIter, nChr, nPop, pc, pm, func, agent_number, Q, Qp, initial_pops):
    """
    Main Program of Non-Dominated Genetic Algorithm
    Params:
        nIter: Number of iterations
        nChr: Chromosome size
        nPop: Population size
        pc: Crossover probability
        pm: Mutation probability
        func: Function to optimize
        agent_number: Number of agents
        Q: Evaluation matrix
        Qp: Preference matrix
    """
    # Generate initial population
    pops = initPops_TAP(nPop, nChr, agent_number, initial_pops)
    fits = fitness_TAP(Q, Qp, pops, func)

    # Start iterations
    iter = 1
    while iter <= nIter:
        print(f"【Progress】【{'▋'*int(iter/nIter*20):20s}】【Generation {iter} in progress...】【Total {nIter} generations】", end='\r')

        ranks = nonDominationSort(pops, fits)  # Non-dominated sorting
        distances = crowdingDistanceSort(pops, fits, ranks)  # Crowding distance
        pops, fits = select1(nPop, pops, fits, ranks, distances)

        chrpops = crossover_GRA(pops, pc, agent_number)  # Crossover to produce offspring
        chrpops = mutate_GRA(chrpops, pm, agent_number)  # Mutation to produce offspring
        chrfits = fitness_TAP(Q, Qp, chrpops, func)

        # Selection from original and offspring populations
        pops, fits = optSelect(pops, fits, chrpops, chrfits)
        iter += 1

    # Final non-dominated sorting
    ranks = nonDominationSort(pops, fits)
    distances = crowdingDistanceSort(pops, fits, ranks)

    paretoPops = pops[ranks == 0].copy()
    paretoFits = fits[ranks == 0].copy()
    paretoPops = np.unique(paretoPops, axis=0).astype(int)
    paretoFits = fitness_TAP(Q, Qp, paretoPops, func)
    original_pops_size = len(pops)
    final_pops = np.unique(pops, axis=0)
    final_fits = fitness_TAP(Q, Qp, final_pops, func)
    return paretoPops, paretoFits, final_pops, final_fits, original_pops_size




