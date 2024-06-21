# THIS FILE IS NOT USED AS THE FINAL IMPLEMENTATION OF SIMULATED ANNEALING, IT IS USED TO FIND OPTIMAL PARAMETERS FOR
# THE ANNEALING PROCESS. IT COMES FROM THE `simanneal` PACKAGE, MAKE SURE TO REFERENCE THAT.

import time

import numpy as np
from simanneal import Annealer

from al.mh.simulated_annealing import SimulatedAnnealing
from al.utils import set_seed
from config import Config
from playground.environment import Environment
from playground.object import TrajectoryObject


class SA(Annealer):
    def __init__(self, state, heuristic, seed):
        super(SA, self).__init__(state)
        self.mh = heuristic

    def move(self):
        while True:
            i = np.random.choice(np.where(self.state == 1)[0])
            if i not in self.mh.locked_subsolution:
                break
        j = np.random.choice(np.where(self.state == 0)[0])
        self.state[i] = 0
        self.state[j] = 1

    def energy(self):
        return -self.mh.evaluate_selection(self.state)

    def update(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    results = []
    start = time.time()

    avg_schedule = dict()

    for s in range(1):
        c = Config()
        c.SEED = s
        set_seed(c.SEED)
        env = Environment(c)
        similarity_dict = env.generate_objects_ail(TrajectoryObject)
        heuristic = SimulatedAnnealing(c, similarity_dict, [])
        sel = heuristic._get_random_initial_selection()
        sa = SA(sel, heuristic, s)
        auto_schedule = sa.auto(minutes=1 / 120)
        for key in auto_schedule.keys():
            if key in avg_schedule.keys():
                avg_schedule[key].append(auto_schedule[key])
            else:
                avg_schedule[key] = [auto_schedule[key]]
        sa.set_schedule(auto_schedule)
        # sa.set_schedule({'tmax': 28.827, 'tmin': 0.1153, 'steps': 20000, "updates": 1})
        state, e = sa.anneal()
        results.append(-e)
    print(f"Mean: {np.mean(results)}, std: {np.std(results)}")
    print(f"Time taken: {time.time() - start}")
    # Mean: 30.18, std: 2.8915278544972267
    # Mean: 30.173333333333332, std: 2.8838554440578714
    print(np.mean(avg_schedule["tmax"]))
    print(np.mean(avg_schedule["tmin"]))
    print(np.mean(avg_schedule["steps"]))
