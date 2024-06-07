import numpy as np
import sklearn.cluster as sc

from config import Config
from environment import Environment
from object import TrajectoryObject


def evaluate_selection(selected_ind, similarities, c):
    objects = set()
    for s in selected_ind:
        # assume similarities[s] returns a sorted list of similarities to all objects
        # where o is an object in objects and s is an object in selected
        # use binary search to find the first object with similarity below threshold
        objs, sim = similarities[s]
        # sim is a sorted list of similarities, use binary search to find the first object with similarity below threshold
        ind = np.searchsorted(sim, c.SIMILARITY_THRESHOLD)
        # print(f"Selected: {s}, threshold: {c.SIMILARITY_THRESHOLD}, ind: {ind}")
        objects.update(objs[ind:])
    return len(objects)


def evaluate_strategy(strategy, c, n=100):
    counts = []
    for i in range(n):
        np.random.seed(i)
        c.TASK_TYPES = ["gripping"]
        c.OBJ_NUM = 100
        c.LATENT_DIM = 10
        env = Environment(c)
        env.generate_objects_ail(TrajectoryObject)
        similarities = env.get_similarities()
        similarity_dict = {}
        for o in env.get_objects():
            s = similarities[o.task_type][o.index]
            ar = []
            for o2 in env.get_objects():
                if o.task_type != o2.task_type:
                    continue
                ar.append((o2.index, s[o2.index]))
            ss = sorted(ar, key=lambda x: x[1])
            similarity_dict[o.index] = ([x[0] for x in ss], [x[1] for x in ss])

        objects = env.get_objects()
        selected = strategy(objects, c, env.get_similarities())
        count = evaluate_selection(selected, similarity_dict, c)
        counts.append(count)
    return np.mean(counts), np.std(counts)


def k_means_strat(objects, c, similarities):
    res = sc.KMeans(n_clusters=c.KNOWN_OBJECT_NUM).fit(np.array([o.visible_repr for o in objects]))
    # select the most central object from each cluster
    selected = []
    for i in range(c.KNOWN_OBJECT_NUM):
        cluster = [o for j, o in enumerate(objects) if res.labels_[j] == i]
        selected.append(min(cluster, key=lambda x: np.linalg.norm(x.visible_repr - res.cluster_centers_[i])))
    return [o.index for o in selected]


def hierarchical_strat(objects, c, similarities):
    res = sc.AgglomerativeClustering(n_clusters=c.KNOWN_OBJECT_NUM).fit(np.array([o.visible_repr for o in objects]))
    # select the most central object from each cluster
    selected = []
    for i in range(c.KNOWN_OBJECT_NUM):
        cluster = [o for j, o in enumerate(objects) if res.labels_[j] == i]
        selected.append(min(cluster, key=lambda x: np.linalg.norm(
            x.visible_repr - np.mean([o.visible_repr for o in cluster], axis=0))))
    return [o.index for o in selected]


if __name__ == "__main__":
    # TODO try to run clustering on this data, both hierarchical (select split equal to budget) and parametric with
    # TODO the param = budget. Try selecting within those clusters
    c = Config()
    random = lambda objects, c, similarities: [o.index for o in np.random.choice(objects, c.KNOWN_OBJECT_NUM, replace=False)]

    print("Random selection")
    mean, std = evaluate_strategy(random, c)
    print(f"Mean: {mean}, std: {std}")

    print("K-means selection")
    mean, std = evaluate_strategy(k_means_strat, c)
    print(f"Mean: {mean}, std: {std}")

    print("Hierarchical selection")
    mean, std = evaluate_strategy(hierarchical_strat, c)
    print(f"Mean: {mean}, std: {std}")
