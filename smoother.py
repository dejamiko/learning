import abc


class Smoother(abc.ABC):
    def __init__(self, c, solver):
        """
        Initialise the smoother with the configuration
        :param c: The configuration
        """
        self.c = c
        self.solver = solver

    @abc.abstractmethod
    def smooth(self, trajectory):
        """
        Smooth the trajectory
        :param trajectory: The trajectory to smooth
        :return: The smoothed trajectory
        """
        pass


class IdentitySmoother(Smoother):
    """
    A smoother that doesn't actually change the trajectory
    """
    def smooth(self, trajectory):
        """
        Return the trajectory as is
        """
        return trajectory


class AverageSmoother(Smoother):
    """
    A smoother that averages the actions to smooth the trajectory
    """
    def __init__(self, c, solver, alpha):
        super().__init__(c, solver)
        self.alpha = alpha

    def smooth(self, trajectory):
        """
        Smooth the trajectory by averaging the actions. To make it less aggressive, use moving values
        :param trajectory: The trajectory to smooth
        :return: The smoothed trajectory
        """
        smoothed_trajectory = []
        for i in range(len(trajectory)):
            if i == 0:
                smoothed_trajectory.append(trajectory[i])
                continue
            val = ((trajectory[i] + trajectory[i - 1]) / 2) * self.alpha + trajectory[i] * (1 - self.alpha)
            smoothed_trajectory.append(val)
        return smoothed_trajectory
