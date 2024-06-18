import numpy as np
import wandb
from pyswarms.discrete import BinaryPSO

from al.utils import MetaHeuristic
from config import Config


class SwarmHeuristic(MetaHeuristic):
    def __init__(self, c, threshold=None):
        super().__init__(c, threshold)

    def cost_function(self, x):
        # this is a cost function which penalises the solutions that select more objects than allowed
        x = np.array(x)
        res = np.apply_along_axis(
            lambda a: (
                np.sum(a)
                if np.sum(a) > self.c.KNOWN_OBJECT_NUM
                else -self.evaluate_selection(a)
            ),
            axis=1,
            arr=x.reshape(-1, self.c.OBJ_NUM),
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
            iters=self.c.MH_BUDGET // self.c.PSO_PARTICLES,
            verbose=False,
        )
        if np.sum(pos) > self.c.KNOWN_OBJECT_NUM:
            return np.zeros(self.c.OBJ_NUM)
        return pos


if __name__ == "__main__":
    config = Config()
    swarm = SwarmHeuristic(config)

    # print(f"Swarm selection for particles={config.PSO_PARTICLES}, c1={config.PSO_C1}, c2={config.PSO_C2}, "
    #       f"w={config.PSO_W}, k={config.PSO_K}, p={config.PSO_P}")
    # mean, std = swarm.evaluate_strategy(n=100)
    # print(f"Mean: {mean}, std: {std}")
    # print(f"Time taken: {swarm.get_mean_time()}")

    wandb.login(key="8d9dd70311672d46669adf913d75468f2ba2095b")

    sweep_config = {
        "name": "Swarm",
        "method": "bayes",
        "metric": {"name": "mean", "goal": "maximize"},
        "parameters": {
            "PSO_PARTICLES": {"min": 10, "max": 200},
            "PSO_C1": {"min": 0.1, "max": 3.0},
            "PSO_C2": {"min": 0.1, "max": 3.0},
            "PSO_W": {"min": 0.1, "max": 3.0},
            "PSO_K": {"min": 10, "max": 200},
            "PSO_P": {"values": [1, 2]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="swarm")


    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            c = Config()
            c.PSO_PARTICLES = config["PSO_PARTICLES"]
            c.PSO_C1 = config["PSO_C1"]
            c.PSO_C2 = config["PSO_C2"]
            c.PSO_W = config["PSO_W"]
            c.PSO_K = config["PSO_K"]
            c.PSO_P = config["PSO_P"]
            wandb.log({"config": c.__dict__})
            print(f"Config: {c.__dict__}")
            if c.PSO_K > c.PSO_PARTICLES:
                return 0

            swarm = SwarmHeuristic(c)
            mean, std = swarm.evaluate_strategy(n=100)
            print(f"Mean: {mean}, std: {std}")
            wandb.log({"mean score": mean})
            return mean


    wandb.agent(sweep_id, train, count=5000)

    wandb.finish()
