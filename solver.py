import time

import numpy as np

from config import Config
from environment import Environment
from oracle import NoisyOracle
from visualiser import Visualiser


def try_object(env, obj_ind, base_trajectory, c):
    """
    Try to solve the object with the given trajectory
    :param env: The environment
    :param obj_ind: The object index
    :param base_trajectory: The base trajectory to try
    :param c: The configuration
    """
    counter = 0
    trajectory = base_trajectory
    failed = not env.try_object_trajectory(obj_ind, trajectory)
    while failed:
        trajectory = base_trajectory + np.random.normal(0, c.ACTION_EXPLORATION_DEVIATION, base_trajectory.shape)
        failed = not env.try_object_trajectory(obj_ind, trajectory)
        counter += 1
        if counter > c.EXPLORATION_TRIES:
            if c.VERBOSITY > 0:
                print(f"Failed to find {env.get_position(obj_ind)} with trajectory ending "
                      f"{env.get_trajectory_end(base_trajectory)} for object {obj_ind}")
            break
    return counter, not failed, trajectory


def solve_objects(env, c):
    """
    Solve all objects in a given environment
    :param env: The environment
    :param c: The configuration
    """
    all_tries = []
    failed_objects = []
    while env.has_unknown_objects():
        u = env.select_object_to_try()

        if c.METHOD == "single":
            demo = env.find_most_similar_known_object_demo(u)
        elif c.METHOD == "average":
            demo = env.find_top_k_most_similar_known_objects_demo(u)
        else:
            raise ValueError(f"Unknown method {c.METHOD}")

        tries, success, trajectory = try_object(env, u, demo, c)

        if not success:
            failed_objects.append(u)
        else:
            all_tries.append(tries)

        env.update_object(u, success, trajectory)

    if c.VERBOSITY > 0:
        print(f"Selection frequency: {env.get_selection_frequency()}")
    return all_tries, failed_objects


def run_experiment(seed, c):
    """
    Run an experiment with a given seed and configuration
    :param seed: The seed
    :param c: The configuration
    """
    start_time = time.time()
    np.random.seed(seed)
    oracle = NoisyOracle(1)

    environment = Environment(c, oracle)
    visualiser = Visualiser(c)

    if c.VERBOSITY > 0:
        visualiser.create_figure_for_all(environment.get_objects(), "_latent_repr")
        visualiser.save_figure("objects_latent.png")
        visualiser.create_figure_for_all(environment.get_objects(), "visible_repr")
        visualiser.save_figure("objects_visible.png")

    all_tries, failed_object_indices = solve_objects(environment, c)

    exploration_tries = all_tries
    exploration_tries_mean = np.mean(exploration_tries) if len(exploration_tries) > 0 else 0
    failures_from_exploration_count = len(failed_object_indices)

    if c.VERBOSITY > 0:
        visualiser.create_figure_for_all(environment.get_objects(), "_latent_repr")
        visualiser.create_figure_for_ind(environment.get_objects(), failed_object_indices, "_latent_repr", "red")
        visualiser.save_figure("objects_latent_with_failures.png")

        visualiser.create_figure_for_all(environment.get_objects(), "visible_repr")
        visualiser.create_figure_for_ind(environment.get_objects(), failed_object_indices, "visible_repr", "red")
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
    c = Config()
    c.METHOD = "single"
    res = run_experiment(0, c)
    c.METHOD = "average"
    c.TOP_K = 1
    res2 = run_experiment(0, c)

    assert res["exploration_tries"] == res2["exploration_tries"]
    assert res["failures_from_exploration_count"] == res2["failures_from_exploration_count"]
