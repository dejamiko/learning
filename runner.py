import numpy as np

from solver import run_experiment

if __name__ == "__main__":
    # TODO Add a cli to specify the configuration

    seeds_to_try = np.arange(10)

    all_results = []
    for seed in seeds_to_try:
        print("Running experiment with seed", seed)
        result = run_experiment(seed)
        all_results.append(result)

    print("Average total cost", np.mean([r["total_cost"] for r in all_results]))
    print("Average oracle tries", np.mean([r["oracle_tries"] for r in all_results]))
    print("Average exploration tries", np.mean([r["exploration_tries"] for r in all_results]))
    print("Average failures from oracle", np.mean([r["failures_from_oracle_count"] for r in all_results]))
    print("Average failures from exploration", np.mean([r["failures_from_exploration_count"] for r in all_results]))
    print("Average time", np.mean([r["time"] for r in all_results]))
