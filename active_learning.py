import numpy as np

from config import Config
from environment import Environment
from object import TrajectoryObject
from oracle import NoisyOracle


def evaluate_selection(objects, selected, similarities, c):
    count = 0
    for o in objects:
        for s in selected:
            if o.task_type == s.task_type:
                if similarities[o.task_type][o.index, s.index] > c.SIMILARITY_THRESHOLD:
                    count += 1
                    break
    return count


if __name__ == "__main__":
    np.random.seed(0)
    c = Config()
    c.TASK_TYPES = ["gripping"]
    c.OBJ_NUM = 100
    c.LATENT_DIM = 10
    oracle = NoisyOracle(1)
    env = Environment(c)
    env.generate_objects_ail(TrajectoryObject)

    objects = env.get_objects()
    counts = []
    for i in range(100):
        selected = np.random.choice(objects, c.KNOWN_OBJECT_NUM, replace=False)
        count = evaluate_selection(objects, selected, env.get_similarities(), c)
        counts.append(count)

    print(f"Average count: {np.mean(counts)}")
    print(f"Standard deviation: {np.std(counts)}")
