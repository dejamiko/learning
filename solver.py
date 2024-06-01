import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

import config as c
from environment import NoisyOracle, TrajectoryObject


def try_object(obj, base_trajectory):
    """
    Try to solve the object by applying the base action with some noise
    :param obj: The object to solve
    :param base_trajectory: The base trajectory to apply
    :return: The number of tries it took to solve the object and whether the object was solved
    """
    counter = 0
    trajectory = base_trajectory
    failed = not obj.try_trajectory(trajectory)
    while failed:
        trajectory = base_trajectory + np.random.normal(0, c.ACTION_EXPLORATION_DEVIATION, base_trajectory.shape)
        failed = not obj.try_trajectory(trajectory)
        counter += 1
        if counter > c.EXPLORATION_TRIES:
            if c.VERBOSITY > 0:
                print(f"Failed to find {obj.get_position()} with trajectory ending {np.sum(base_trajectory, axis=0)} "
                      f"for object {obj}")
            break
    return counter, not failed, trajectory


def select_objects_to_try(visual_similarities, known_objects, unknown_objects, object_types, objects):
    task_types = np.unique(object_types[known_objects])
    candidates = []
    for t_t in task_types:
        # find the known and unknown objects that have the task type t_t
        known_objects_t = known_objects[object_types[known_objects] == t_t]
        # filter out known objects where the demo is None
        known_objects_t = known_objects_t[[objects[i].demo is not None for i in known_objects_t]]
        unknown_objects_t = unknown_objects[object_types[unknown_objects] == t_t]
        if len(known_objects_t) == 0 or len(unknown_objects_t) == 0:
            continue
        # find the object from the unknown objects that is most similar to one of the known objects
        u = unknown_objects_t[np.argsort(np.max(visual_similarities[unknown_objects_t][:, known_objects_t], axis=1))][
            -1]
        # find the corresponding known object taking into account the task type
        k = known_objects_t[np.argmax(visual_similarities[u, known_objects_t])]
        candidates.append((u, k))
        if c.VERBOSITY > 1:
            print(f"Task type {t_t}, unknown object {u}, known object {k}, similarity {visual_similarities[u, k]}")

    # find the pair of objects that are most similar
    u, k = candidates[np.argmax([visual_similarities[u, k] for u, k in candidates])]
    if c.VERBOSITY > 1:
        print(f"Selected unknown object {u}, known object {k}, similarity {visual_similarities[u, k]}")
    return u, k


def create_figure(objects, field):
    to_plot = {t: [] for t in c.TASK_TYPES}
    for i, o in enumerate(objects):
        to_plot[o.task_type].append(getattr(o, field))
    for t in c.TASK_TYPES:
        to_plot[t] = np.array(to_plot[t])
        plt.scatter(to_plot[t][:, 0], to_plot[t][:, 1], label=t)


def generate_objects():
    objects = []
    for i in range(c.OBJ_NUM):
        objects.append(TrajectoryObject(i, np.random.uniform(0, 1, c.LATENT_DIM), -1))
    # cluster the objects into different task types based on the latent representation
    k = KMeans(n_clusters=len(c.TASK_TYPES)).fit(np.array([o._latent_repr for o in objects]))
    for i, o in enumerate(objects):
        o.task_type = c.TASK_TYPES[k.labels_[i]]
        o.generate_waypoints()

    create_figure(objects, "_latent_repr")
    plt.savefig("objects_latent.png")
    plt.clf()
    create_figure(objects, "visible_repr")
    plt.savefig("objects_visible.png")
    plt.clf()

    return np.array(objects, dtype=TrajectoryObject)


def generate_helper_data(objects, oracle):
    # find random objects to be provided as known but make sure all task types are represented
    known_objects = np.random.choice(c.OBJ_NUM, c.KNOWN_OBJECT_NUM, replace=False)
    while len(set([objects[i].task_type for i in known_objects])) < len(c.TASK_TYPES):
        if c.VERBOSITY > 0:
            print("Retrying object selection to ensure all task types are represented")
        known_objects = np.random.choice(c.OBJ_NUM, c.KNOWN_OBJECT_NUM, replace=False)

    for i in known_objects:
        objects[i].demo = oracle.get_demo(objects[i])
    oracle.reset_cost()

    unknown_objects = np.array([i for i in range(c.OBJ_NUM) if i not in known_objects])

    visual_similarities = np.zeros((c.OBJ_NUM, c.OBJ_NUM))
    for i in range(c.OBJ_NUM):
        for j in range(c.OBJ_NUM):
            visual_similarities[i, j] = objects[i].get_visual_similarity(objects[j])

    # apply min-max normalization to the visual similarities
    min_ = -1
    max_ = 1
    visual_similarities = (visual_similarities - min_) / (max_ - min_)

    object_types = [objects[i].task_type for i in range(c.OBJ_NUM)]
    object_types = np.array(object_types)

    return known_objects, unknown_objects, visual_similarities, object_types


def solve_objects(objects, known_objects, unknown_objects, visual_similarities, object_types, oracle):
    all_tries = []
    failed_objects = []
    selection_frequency = np.zeros(c.OBJ_NUM)
    while len(known_objects) < c.OBJ_NUM:
        u, k = select_objects_to_try(visual_similarities, known_objects, unknown_objects, object_types, objects)
        selection_frequency[k] += 1
        # if the objects are quite similar, try replaying the known demo
        if c.VERBOSITY > 0 and not objects[u].get_task_type_correspondence(objects[k]):
            print(f"Similarity {visual_similarities[u, k]}, matching tasks "
                  f"{objects[u].get_task_type_correspondence(objects[k])}")
        if c.VERBOSITY > 1:
            print(f"Replaying demo for object {objects[k]} to solve object {objects[u]} with similarity "
                  f"{visual_similarities[u, k]}")
        demo = objects[k].demo

        tries, success, trajectory = try_object(objects[u], demo)

        if not success:
            failed_objects.append(u)
        else:
            objects[u].demo = trajectory
            all_tries.append(tries)

        known_objects = np.append(known_objects, u)
        unknown_objects = np.array([i for i in range(c.OBJ_NUM) if i not in known_objects])
    if c.VERBOSITY > 0:
        print(f"Selection frequency: {selection_frequency}")
    return all_tries, failed_objects


def run_experiment(seed):
    start_time = time.time()
    np.random.seed(seed)
    oracle = NoisyOracle(1)

    objects = generate_objects()
    known_objects, unknown_objects, visual_similarities, object_types = generate_helper_data(objects, oracle)

    all_tries, failed_objects = solve_objects(objects, known_objects, unknown_objects, visual_similarities,
                                              object_types, oracle)

    total_cost = oracle.get_final_cost()
    exploration_tries = all_tries
    exploration_tries_mean = np.mean(exploration_tries) if len(exploration_tries) > 0 else 0
    failures_from_exploration_count = len([o for o in failed_objects])

    if c.VERBOSITY > 0:
        create_figure(objects, "_latent_repr")
        for obj_ind in failed_objects:
            plt.scatter(objects[obj_ind]._latent_repr[0], objects[obj_ind]._latent_repr[1], color="red")
        plt.savefig("objects_with_failures.png")
        plt.clf()
        create_figure(objects, "visible_repr")
        for obj_ind in failed_objects:
            plt.scatter(objects[obj_ind].visible_repr[0], objects[obj_ind].visible_repr[1], color="red")
        plt.savefig("objects_visible_with_failures.png")

        print(f"Average tries with exploration: {np.mean(exploration_tries)}")
        print(f"Standard deviation of tries with exploration: {np.std(exploration_tries)}")
        print(f"Number of failures from exploration: {failures_from_exploration_count}")
        s_exploration_tries = np.sort(exploration_tries)
        print(f"Exploration tries: {s_exploration_tries}")

    return {
        "seed": seed,
        "total_cost": total_cost,
        "exploration_tries": exploration_tries_mean,
        "failures_from_exploration_count": failures_from_exploration_count,
        "time": time.time() - start_time
    }


if __name__ == "__main__":
    # TODO Add learning so the exploration is smarter - how? Do I have a reward signal?
    run_experiment(0)
