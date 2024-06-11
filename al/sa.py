import numpy as np

from al.utils import MetaHeuristic
from config import Config


class SimulatedAnnealing(MetaHeuristic):
    def __init__(self, c, threshold=None):
        super().__init__(c, threshold)

    def strategy(self):
        selected = self._get_initial_selection()
        self.best_selection = selected
        best_cost = self.get_cost(selected)
        T = self.c.SA_T

        while self.count < self.c.MH_BUDGET:
            new_selection = self.get_random_neighbour(selected)
            new_cost = self.get_cost(new_selection)

            diff = best_cost - new_cost

            if diff >= 0:
                self.best_selection = new_selection
            elif np.random.uniform() <= np.exp(diff / T):
                self.best_selection = new_selection

            T = T * self.c.SA_ALPHA
            T = max(T, self.c.SA_T_MIN)
            selected = new_selection
            if new_cost < best_cost:
                best_cost = new_cost
                self.best_selection = new_selection
        return self.best_selection


if __name__ == "__main__":
    c = Config()

    sa = SimulatedAnnealing(c)

    print(f"Simulated annealing selection for alpha={c.SA_ALPHA}, T={c.SA_T}")
    mean, std = sa.evaluate_strategy(n=5)
    print(f"Mean: {mean}, std: {std}")

    for t in range(1, 100, 10):
        for alpha in np.arange(0.8, 1, 0.01):
            c.SA_T = t
            c.SA_ALPHA = alpha

            print(f"Simulated annealing selection for alpha={alpha}, T={t}")
            mean, std = sa.evaluate_strategy(n=5)
            print(f"Mean: {mean}, std: {std}, time taken: {sa.get_mean_time()}")
