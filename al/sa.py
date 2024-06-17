import numpy as np

from al.utils import MetaHeuristic
from config import Config


class SimulatedAnnealing(MetaHeuristic):
    def __init__(self, c, threshold=None):
        super().__init__(c, threshold)

    def strategy(self):
        selection = self._get_initial_selection()
        score = self.evaluate_selection(selection)
        self.best_selection = selection.copy()
        best_score = score
        prev_state = selection.copy()
        prev_score = score
        t_factor = -np.log(self.c.SA_T) / self.c.MH_BUDGET

        while self.count < self.c.MH_BUDGET + 1:
            T = self.c.SA_T * np.exp(t_factor * (self.count - 1) / self.c.MH_BUDGET)
            selection = self.get_random_neighbour(selection)
            score = self.evaluate_selection(selection)
            diff = score - prev_score
            if diff < 0.0 and np.exp(diff / T) < np.random.rand():
                # Restore previous state
                selection = prev_state.copy()
            else:
                # Accept new state
                prev_state = selection.copy()
                prev_score = score
                if score > best_score:
                    best_score = score
                    self.best_selection = selection.copy()
        return self.best_selection


if __name__ == "__main__":
    c = Config()
    # c.SA_T = 21.836734693877553
    # c.SA_T_MIN = 0.08673469387755103

    sa = SimulatedAnnealing(c)

    print(f"Simulated annealing selection for T={c.SA_T}, T_MIN={c.SA_T_MIN}")
    mean, std = sa.evaluate_strategy(n=100)
    print(f"Mean: {mean}, std: {std}")
    print(f"Time taken: {sa.get_mean_time()}")

    # results = []
    #
    # for t_max in np.linspace(20, 50, 50):
    #     for t_min in np.linspace(0.05, 0.5, 50):
    #         c.SA_T = t_max
    #         c.SA_T_MIN = t_min
    #         sa = SimulatedAnnealing(c)
    #         mean, std = sa.evaluate_strategy(n=10)
    #         results.append((t_max, t_min, mean, std))
    #         print(f"Simulated annealing selection for T={c.SA_T}, T_MIN={c.SA_T_MIN} - Mean: {mean}, std: {std}")
    #
    # sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    # # print the best 10 combinations
    # for i in range(10):
    #     print(f"Best selection for T={sorted_results[i][0]}, T_MIN={sorted_results[i][1]} - "
    #           f"Mean: {sorted_results[i][2]}, std: {sorted_results[i][3]}")

