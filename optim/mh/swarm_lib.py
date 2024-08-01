import numpy as np
from pyswarms.discrete import BinaryPSO

from config import Config
from optim.mh.metaheuristic import MetaHeuristic
from playground.environment import Environment


class SwarmHeuristic(MetaHeuristic):
    def __init__(self, c, similarity_dict, locked_subsolution):
        super().__init__(c, similarity_dict, locked_subsolution)
        self.best_selection = None
        self.optimiser = None
        np.random.seed(c.SEED)

    def get_cost_for_selection(self, selection):
        selection[self.locked_subsolution] = 1
        return -self.evaluate_selection_with_constraint_penalty(selection)

    def cost_function(self, x):
        x = np.array(x)
        res = np.apply_along_axis(
            self.get_cost_for_selection, axis=1, arr=x.reshape(-1, self.c.OBJ_NUM)
        )
        return res

    def strategy(self):
        options = {
            "c1": self.c.PSO_C1,
            "c2": self.c.PSO_C2,
            "w": self.c.PSO_W,
            "k": self.c.PSO_K,
            "p": self.c.PSO_P,
        }
        self.optimiser = BinaryPSO(
            n_particles=self.c.PSO_PARTICLES, dimensions=self.c.OBJ_NUM, options=options
        )
        cost, pos = self.optimiser.optimize(
            self.cost_function,
            iters=int(self.c.MH_BUDGET / self.c.PSO_PARTICLES),
            verbose=False,
        )
        if np.sum(pos) != self.c.DEMONSTRATION_BUDGET:
            self.best_selection = self.get_random_selection()
        else:
            self.best_selection = pos
        return self.best_selection

    def get_best_solution(self):
        if self.best_selection is not None:
            return self.best_selection
        final_best_pos = self.optimiser.swarm.pbest_pos[
            self.optimiser.swarm.pbest_cost.argmin()
        ].copy()
        if np.sum(final_best_pos) != self.c.DEMONSTRATION_BUDGET:
            return self.get_random_selection()
        return final_best_pos


if __name__ == "__main__":
    config = Config()
    env = Environment(config)
    swarm = SwarmHeuristic(config, env, [])

    selected = swarm.strategy()
    print(swarm.evaluate_selection_with_constraint_penalty(selected))
