import time

import numpy as np


class Object:
    def __init__(self, latent_value, index):
        self.latent_value = latent_value
        self.index = index

    def __str__(self):
        return f"Object {self.index}: {self.latent_value}"

    def __repr__(self):
        return self.__str__()


def compute_dist(o1: Object, o2: Object):
    return np.linalg.norm(o1.latent_value - o2.latent_value)


def try_grasp(dist_to_other):
    # make the probability of grasping dependent on the distance between the two objects
    # prob = 0.9 / (np.exp(100 * dist_to_other))  # some function of the distance that decays quickly
    prob = (
        -5 * dist_to_other + 0.9
    )  # a linear probability function for which greedy should be optimal
    return np.random.random() < prob


def get_prob_from_dist(dist):
    return -5 * dist + 0.9


def evaluate_order(
    unlabelled_object_indices, labelled_object_indices, distances, method, active=False
):
    # all_tries = 0
    sum_of_dist = 0
    labelled_object_indices_cp = labelled_object_indices.copy()
    while len(unlabelled_object_indices) > 0:
        # find the closest labelled object
        unlabelled_object_index = unlabelled_object_indices[0]
        # remove the object from the unlabelled set
        unlabelled_object_indices = unlabelled_object_indices[1:]
        dist_to_closest = np.min(
            distances[unlabelled_object_index][labelled_object_indices_cp]
        )
        labelled_object_indices_cp = np.concatenate(
            (labelled_object_indices_cp, [unlabelled_object_index])
        )
        unlabelled_object_indices_new = reorder(
            unlabelled_object_indices,
            labelled_object_indices_cp,
            distances,
            method,
            active,
        )
        assert len(unlabelled_object_indices_new) == len(unlabelled_object_indices)
        unlabelled_object_indices = unlabelled_object_indices_new
        sum_of_dist += dist_to_closest
    return sum_of_dist


def reorder(
    unlabelled_object_indices, labelled_object_indices, distances, method, active=False
):
    if not active or len(unlabelled_object_indices) < 2:
        return unlabelled_object_indices
    if method == "random":
        return np.random.permutation(unlabelled_object_indices)
    elif method == "closest":
        return unlabelled_object_indices[
            np.argsort(
                np.min(
                    distances[unlabelled_object_indices][:, labelled_object_indices],
                    axis=1,
                )
            )
        ]
    elif method == "furthest":
        return unlabelled_object_indices[
            np.argsort(
                -np.min(
                    distances[unlabelled_object_indices][:, labelled_object_indices],
                    axis=1,
                )
            )
        ]
    elif method == "balanced":
        # find the k closest objects
        k = 25
        closest_indices = unlabelled_object_indices[
            np.argsort(
                np.min(
                    distances[unlabelled_object_indices][:, labelled_object_indices],
                    axis=1,
                )
            )
        ][0:k]
        # out of those k, find one that's closest to the other unlabelled objects on average
        dist_sum = np.ones_like(closest_indices)
        for i, c in enumerate(closest_indices):
            dist_sum[i] = np.sum(distances[unlabelled_object_indices][:c])
        min_ind = np.argmin(dist_sum)

        return np.concatenate(
            (
                [closest_indices[min_ind]],
                unlabelled_object_indices[
                    unlabelled_object_indices != closest_indices[min_ind]
                ],
            )
        )
    elif method == "beam_search":
        k = 25
        sequences = []
        scores = []
        min_dist = np.min(
            distances[unlabelled_object_indices][:, labelled_object_indices], axis=1
        )
        sort_indices = np.argsort(min_dist)
        first = unlabelled_object_indices[sort_indices][:k]
        min_dist = min_dist[sort_indices][:k]
        for f, d in zip(first, min_dist):
            sequences.append([f])
            scores.append(d)

        for _ in range(len(unlabelled_object_indices) - 1):
            new_ss = []
            new_ws = np.zeros(k)
            for s, w in zip(sequences, scores):
                # expand the sequence by adding each unlabelled object
                for u in unlabelled_object_indices:
                    if u not in s:
                        if len(new_ss) < k:
                            new_ss.append(s + [u])
                            new_ws[len(new_ss) - 1] = w + np.min(
                                distances[u][
                                    np.concatenate((labelled_object_indices, s))
                                ]
                            )
                        else:
                            # find the largest and replace it
                            max_ind = np.argmax(new_ws)
                            new_ss[max_ind] = s + [u]
                            new_ws[max_ind] = w + np.min(
                                distances[u][
                                    np.concatenate((labelled_object_indices, s))
                                ]
                            )
            sequences = new_ss
            scores = new_ws

        min_ind = np.argmin(scores)
        return sequences[min_ind]
    else:
        raise ValueError("Invalid method")


def run_test(method, iterative=False):
    random_seeds = list(range(100))
    num_objects = 100
    num_labelled = 10
    all_dist = []
    times = []
    for seed in random_seeds:
        start = time.time()
        np.random.seed(seed)
        objects = [
            Object(np.array((np.random.random(), np.random.random())), i)
            for i in range(num_objects)
        ]

        # find the distances between any two objects
        distances = np.zeros((num_objects, num_objects))
        for i, o1 in enumerate(objects):
            for j, o2 in enumerate(objects):
                distances[i, j] = compute_dist(o1, o2)

        # rescale the distances by max and min
        distances = (distances - distances.min()) / (distances.max() - distances.min())

        # mark some objects as labelled
        labelled_object_indices = np.random.choice(
            num_objects, num_labelled, replace=False
        )
        unlabelled_object_indices = np.array(
            [i for i in range(num_objects) if i not in labelled_object_indices]
        )

        # the first reorder is always active
        unlabelled_object_indices = reorder(
            unlabelled_object_indices,
            labelled_object_indices,
            distances,
            method,
            active=True,
        )

        all_dist.append(
            evaluate_order(
                unlabelled_object_indices,
                labelled_object_indices,
                distances,
                method,
                iterative,
            )
        )
        times.append(time.time() - start)
    print(f"Method: {method}")
    print(f"Average dist: {np.mean(all_dist)}")
    print(f"Standard deviation: {np.std(all_dist)}")
    print(f"Average time: {np.mean(times)}")


if __name__ == "__main__":
    run_test("random")
    run_test("closest")
    run_test("furthest")
    run_test("balanced")
    run_test("beam_search")
    run_test("closest", iterative=True)
    run_test("furthest", iterative=True)
    run_test("balanced", iterative=True)
