import abc

import numpy as np


class Oracle(abc.ABC):
    """
    An abstract class for an oracle that can provide demonstrations

    The oracle has a cost associated with using it and can provide a demonstration for an object
    """

    def __init__(self, cost):
        """
        Initialize the oracle with the cost per demonstration
        :param cost: The cost per demonstration
        """
        self.cost_per = cost
        self.total_cost = 0

    def get_final_cost(self):
        """
        Get the total cost of using the oracle
        :return: The total cost of using the oracle
        """
        return self.total_cost

    def reset_cost(self):
        """
        Reset the total cost of using the oracle
        """
        self.total_cost = 0

    @abc.abstractmethod
    def get_demo(self, env, obj, c):
        """
        Get a demonstration for the object
        :param env: The environment
        :param obj: The object
        :param c: The configuration
        """
        pass


class NoisyOracle(Oracle):
    """
    A noisy oracle that provides noisy demonstrations
    """

    def get_demo(self, env, obj, c):
        """
        Get a demonstration for the object, adding noise to the demonstration
        :param env: The environment
        :param obj: The object
        :param c: The configuration
        :return: The demonstration for the object
        """
        self.total_cost += self.cost_per
        obj_demo = env.generate_trajectory(obj.waypoints)
        return obj_demo + np.random.normal(0, c.DEMO_NOISE, obj_demo.shape)


class PerfectOracle(Oracle):
    """
    A perfect oracle that provides perfect demonstrations
    """

    def get_demo(self, env, obj, c):
        """
        Get a demonstration for the object
        :param env: The environment
        :param obj: The object
        :param c: The configuration
        :return: The demonstration for the object
        """
        self.total_cost += self.cost_per
        return env.generate_trajectory(obj.waypoints)
