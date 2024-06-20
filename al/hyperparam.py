import mealpy
import numpy as np

from al.mh import EvolutionaryStrategy, TabuSearch
from al.mh.mealpy_lib import MealpyHeuristic
from al.mh.simulated_annealing import SimulatedAnnealing
from al.solver import evaluate_heuristic, Solver
from config import Config


def find_es_hyperparameters(config):
    results = []
    for pop_size in np.linspace(50, 300, 5):
        for mutation_rate in np.linspace(0.01, 0.3, 5):
            for elite_prop in np.linspace(0.01, 0.3, 5):
                pop_size = int(pop_size)
                config.ES_POP_SIZE = pop_size
                config.ES_MUTATION_RATE = mutation_rate
                config.ES_ELITE_PROP = elite_prop

                mean, std, time = evaluate_heuristic(
                    Solver, config, EvolutionaryStrategy, n=50
                )
                results.append((pop_size, mutation_rate, elite_prop, mean, std, time))

    results = sorted(results, key=lambda x: x[3], reverse=True)
    for result in results[:5]:
        print(
            f"Population size: {result[0]}, Mutation rate: {result[1]}, Elite proportion: {result[2]}, "
            f"Mean: {result[3]} +/- {result[4]}, Time: {result[5]}"
        )


def find_mealpy_optimiser(config):
    results = []

    for optimizer_name in mealpy.get_all_optimizers().keys():
        try:
            config.MP_OPTIMISER_NAME = optimizer_name
            mean, std, time = evaluate_heuristic(Solver, config, MealpyHeuristic, n=100)
            if mean > 0 and time < 1.0:
                results.append((optimizer_name, mean, std, time))
        except Exception:
            continue

    results.sort(key=lambda x: x[1], reverse=True)
    for name, mean, std, time in results:
        print(f"{name}: {mean} +/- {std}, time: {time}")


def evaluate_mealpy_optimisers(config):
    results = []

    for optimizer_name in [
        "DevSPBO",
        "OriginalWarSO",
        "OriginalWaOA",
        "DevSCA",
        "OriginalServalOA",
        "OriginalNGO",
        "OriginalCoatiOA",
        "AdaptiveBA",
        "ImprovedSFO",
        "OriginalGTO",
        "DevBRO",
        "OriginalFFA",
    ]:
        config.MP_OPTIMISER_NAME = optimizer_name
        mean, std, time = evaluate_heuristic(Solver, config, MealpyHeuristic, n=500)
        results.append((optimizer_name, mean, std, time))
    results.sort(key=lambda x: x[1], reverse=True)
    for name, mean, std, time in results:
        print(f"{name}: {mean} +/- {std}, time: {time}")


def find_sa_hyperparameters(config):
    # c.SA_T = 21.836734693877553
    # c.SA_T_MIN = 0.08673469387755103
    results = []

    for t_max in np.linspace(20, 50, 50):
        for t_min in np.linspace(0.05, 0.5, 50):
            config.SA_T = t_max
            config.SA_T_MIN = t_min

            mean, std, time = evaluate_heuristic(
                Solver, config, SimulatedAnnealing, n=100
            )
            results.append((t_max, t_min, mean, std, time))

    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    # print the best 10 combinations
    for t_max, t_min, mean, std, time in sorted_results[:10]:
        print(f"T_max: {t_max}, T_min: {t_min}, Mean: {mean} +/- {std}, Time: {time}")


def find_ts_hyperparameters(config):
    for tabu_list_size in [10, 100, 1000, 10000, 100000]:
        config.TS_L = tabu_list_size
        print(f"Tabu search selection for L={config.TS_L}")
        mean, std, time = evaluate_heuristic(Solver, config, TabuSearch, n=10)
        print(f"Mean: {mean} +/- {std}, Time: {time}")


if __name__ == "__main__":
    c = Config()
    c.TASK_TYPES = ["sample task"]
    find_es_hyperparameters(c)
    find_mealpy_optimiser(c)
    find_sa_hyperparameters(c)
    find_ts_hyperparameters(c)
    evaluate_mealpy_optimisers(c)
