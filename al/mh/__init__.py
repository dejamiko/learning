from al.mh.evolutionary_strategy import EvolutionaryStrategy
from al.mh.mealpy_lib import MealpyHeuristic
from al.mh.randomised_hill_climbing import RandomisedHillClimbing
from al.mh.simulated_annealing import SimulatedAnnealing
# from al.mh.swarm_lib import SwarmHeuristic
from al.mh.tabu_search import TabuSearch


def get_all_heuristics():
    return [
        EvolutionaryStrategy,
        RandomisedHillClimbing,
        SimulatedAnnealing,
        TabuSearch,
        # SwarmHeuristic,
        MealpyHeuristic,
    ]
