import numpy as np

from al.mh.metaheuristic import MetaHeuristic
from al.utils import set_seed
from config import Config
from playground.environment import Environment


class SimulatedAnnealing(MetaHeuristic):
    def __init__(self, c, similarity_dict, locked_subsolution, threshold=None):
        super().__init__(c, similarity_dict, locked_subsolution, threshold)
        self.best_selection = None

    def _get_random_neighbour(self, selected):
        # find a random index where selected is 1 and another where it is 0
        while True:
            i = np.random.choice(np.where(selected == 1)[0])
            if i not in self.locked_subsolution:
                break
        j = np.random.choice(np.where(selected == 0)[0])
        new_selected = selected.copy()
        new_selected[i] = 0
        new_selected[j] = 1
        return new_selected

    def strategy(self):
        selection = self.get_random_initial_selection()
        score = self.evaluate_selection(selection)
        self.best_selection = selection.copy()
        best_score = score
        prev_state = selection.copy()
        prev_score = score
        t_factor = -np.log(self.c.SA_T) / self.c.MH_BUDGET

        while self.count < self.c.MH_BUDGET + 1:
            T = self.c.SA_T * np.exp(t_factor * (self.count - 1) / self.c.MH_BUDGET)
            selection = self._get_random_neighbour(selection)
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

    def get_best_solution(self):
        return self.best_selection


if __name__ == "__main__":
    c = Config()
    set_seed(c.SEED)
    env = Environment(c)
    sa = SimulatedAnnealing(c, env, [])

    selected = sa.strategy()
    print(sa.evaluate_selection(selected))
