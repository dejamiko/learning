import time

import numpy as np

from config import Config
from environment import Environment
from object import TrajectoryObject
from oracle import NoisyOracle
from smoother import IdentitySmoother
from visualiser import Visualiser


class Solver:
    """
    A class to solve the objects in an environment
    """

    def __init__(self, c):
        self.c = c
        self.env = None
        self.smoother = None

    def try_object(self, obj_ind, base_trajectory):
        """
        Try to solve the object with the given trajectory
        :param obj_ind: The object index
        :param base_trajectory: The base trajectory to try
        """
        counter = 0
        trajectory = base_trajectory
        failed = not self.env.try_object_trajectory(obj_ind, trajectory)
        while failed:
            trajectory = base_trajectory + np.random.normal(0, self.c.ACTION_EXPLORATION_DEVIATION,
                                                            base_trajectory.shape)
            failed = not self.env.try_object_trajectory(obj_ind, trajectory)
            counter += 1
            if counter > self.c.EXPLORATION_TRIES:
                if self.c.VERBOSITY > 0:
                    print(f"Failed to find {self.env.get_position(obj_ind)} with trajectory ending "
                          f"{self.env.get_trajectory_end(base_trajectory)} for object {obj_ind}")
                break
        return counter, not failed, trajectory

    def solve_objects(self):
        """
        Solve all objects in a given environment
        """
        all_tries = []
        failed_objects = []
        while self.env.has_unknown_objects():
            u = self.env.select_object_to_try()

            if self.c.METHOD == "single":
                demo = self.env.find_most_similar_known_object_demo(u)
            elif self.c.METHOD == "average":
                demo = self.env.find_top_k_most_similar_known_objects_demo(u)
            else:
                raise ValueError(f"Unknown method {self.c.METHOD}")

            tries, success, trajectory = self.try_object(u, demo)

            if not success:
                failed_objects.append(u)
            else:
                all_tries.append(tries)

            if self.smoother is not None:
                trajectory = self.iterative_smooth(u, trajectory)

            self.env.update_object(u, success, trajectory)

        if self.c.VERBOSITY > 0:
            print(f"Selection frequency: {self.env.get_selection_frequency()}")
        return all_tries, failed_objects

    def iterative_smooth(self, u, trajectory):
        """
        Perform iterative smoothing on the trajectory
        """
        new_trajectory = trajectory
        counter = 0
        while self.env.try_object_trajectory(u, new_trajectory) and counter < self.c.SMOOTHING_TRIES:
            new_trajectory = self.smoother.smooth(new_trajectory)
            counter += 1
            if self.c.VERBOSITY > 1:
                print(f"Smoothing trajectory for object {u}")
        return new_trajectory

    def run_experiment(self, seed):
        """
        Run an experiment with a given seed and configuration
        :param seed: The seed
        """
        start_time = time.time()
        np.random.seed(seed)
        oracle = NoisyOracle(1)

        self.env = Environment(self.c)
        self.env.generate_objects(TrajectoryObject, oracle)
        visualiser = Visualiser(self.c)

        if self.c.VERBOSITY > 0:
            visualiser.create_figure_for_all(self.env.get_objects(), "_latent_repr")
            visualiser.save_figure("objects_latent.png")
            visualiser.create_figure_for_all(self.env.get_objects(), "visible_repr")
            visualiser.save_figure("objects_visible.png")

        all_tries, failed_object_indices = self.solve_objects()

        exploration_tries = all_tries
        exploration_tries_mean = np.mean(exploration_tries) if len(exploration_tries) > 0 else 0
        failures_from_exploration_count = len(failed_object_indices)

        if self.c.VERBOSITY > 0:
            visualiser.create_figure_for_all(self.env.get_objects(), "_latent_repr")
            visualiser.create_figure_for_ind(self.env.get_objects(), failed_object_indices, "_latent_repr", "red")
            visualiser.save_figure("objects_latent_with_failures.png")

            visualiser.create_figure_for_all(self.env.get_objects(), "visible_repr")
            visualiser.create_figure_for_ind(self.env.get_objects(), failed_object_indices, "visible_repr", "red")
            visualiser.save_figure("objects_visible_with_failures.png")

            print(f"Average tries with exploration: {np.mean(exploration_tries)}")
            print(f"Standard deviation of tries with exploration: {np.std(exploration_tries)}")
            print(f"Number of failures from exploration: {failures_from_exploration_count}")
            s_exploration_tries = np.sort(exploration_tries)
            print(f"Exploration tries: {s_exploration_tries}")

        return {
            "seed": seed,
            "exploration_tries": exploration_tries_mean,
            "failures_from_exploration_count": failures_from_exploration_count,
            "time": time.time() - start_time
        }


if __name__ == "__main__":
    # TODO something clever in storage.update_objects - potentially keep the object as unknown but wait to try it again
    conf = Config()
    conf.METHOD = "single"
    solver = Solver(conf)
    solver.smoother = IdentitySmoother(conf, solver)
    res = solver.run_experiment(0)
    conf.METHOD = "average"
    conf.TOP_K = 1
    res2 = solver.run_experiment(0)

    assert res["exploration_tries"] == res2["exploration_tries"]
    assert res["failures_from_exploration_count"] == res2["failures_from_exploration_count"]
