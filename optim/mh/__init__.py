from optim.mh.clustering import ClusteringSearch
from optim.mh.evolutionary_strategy import EvolutionaryStrategy
from optim.mh.exhaustive_search import ExhaustiveSearch
from optim.mh.greedy_local import GreedyLocalSearch
from optim.mh.mealpy_lib import MealpyHeuristic
from optim.mh.random_search import RandomSearchIter, RandomSearch
from optim.mh.randomised_hill_climbing import RandomisedHillClimbing
from optim.mh.simulated_annealing import SimulatedAnnealing
from optim.mh.swarm_lib import SwarmHeuristic
from optim.mh.tabu_search import TabuSearch


def get_all_heuristics():
    return [
        # baselines
        RandomSearch,
        ClusteringSearch,
        GreedyLocalSearch,
        RandomSearchIter,
        ExhaustiveSearch,
        # metaheuristics
        EvolutionaryStrategy,
        RandomisedHillClimbing,
        SimulatedAnnealing,
        TabuSearch,
        SwarmHeuristic,
        MealpyHeuristic,
    ]
