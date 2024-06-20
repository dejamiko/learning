import numpy as np
from pyswarms.discrete import BinaryPSO

from al.mh.metaheuristic import MetaHeuristic
from al.utils import set_seed
from config import Config


class SwarmHeuristic(MetaHeuristic):
    def __init__(self, c, threshold=None):
        super().__init__(c, threshold)

    def get_cost_for_selection(self, selection):
        selection[self.locked_subsolution] = 1
        return -self.evaluate_selection_with_constraint_penalty(selection)

    def cost_function(self, x):
        x = np.array(x)
        res = np.apply_along_axis(
            self.get_cost_for_selection,
            axis=1,
            arr=x.reshape(-1, self.c.OBJ_NUM)
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
        optimizer = BinaryPSO(
            n_particles=self.c.PSO_PARTICLES, dimensions=self.c.OBJ_NUM, options=options
        )
        cost, pos = optimizer.optimize(
            self.cost_function,
            iters=int(self.c.MH_BUDGET / self.c.PSO_PARTICLES),
            verbose=False,
        )
        if np.sum(pos) > self.c.KNOWN_OBJECT_NUM:
            return np.zeros(self.c.OBJ_NUM)
        return pos


if __name__ == "__main__":
    config = Config()
    set_seed(config.SEED)
    swarm = SwarmHeuristic(config)

    swarm.initialise_data()
    selected = swarm.strategy()
    print(swarm.evaluate_selection_with_constraint_penalty(selected))
