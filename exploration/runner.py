import numpy as np

from config import Config
from solver import Solver


def run_exp_on_seeds(seeds_to_try, c):
    all_results = []
    solver = Solver(c)
    for seed in seeds_to_try:
        print("Running experiment with seed", seed)
        result = solver.run_experiment(seed)
        all_results.append(result)

    for r in all_results:
        for k, v in r.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    # TODO Add a cli to specify the configuration
    c = Config()
    c.VERBOSITY = 0
    seeds_to_try = np.arange(1)

    for k in range(1, 10):
        c.TOP_K = k
        run_exp_on_seeds(seeds_to_try, c)
        print(f"For config {c}")
