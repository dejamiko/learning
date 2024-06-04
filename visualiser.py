import matplotlib.pyplot as plt
import numpy as np


class Visualiser:
    """
    A class to create visualisations of the objects
    """

    def __init__(self, c):
        """
        Initialise the visualiser with the configuration
        :param c: The configuration
        """
        self.c = c

    def create_figure_for_all(self, objects, field):
        """
        Create a figure for all objects provided using the field provided
        :param objects: The objects to plot
        :param field: The object field to extract
        """
        to_plot = {t: [] for t in self.c.TASK_TYPES}
        for i, o in enumerate(objects):
            to_plot[o.task_type].append(getattr(o, field))
        for t in self.c.TASK_TYPES:
            to_plot[t] = np.array(to_plot[t])
            plt.scatter(to_plot[t][:, 0], to_plot[t][:, 1], label=t)

    def create_figure_for_ind(self, objects, indices, field, colour=None):
        """
        Create a figure for the objects at the indices provided using the field provided
        :param objects: The objects to plot
        :param indices: The indices of the objects to plot
        :param field: The object field to extract
        :param colour: (Optional) The colour to use for the plot
        """
        for i in indices:
            if colour is not None:
                plt.scatter(getattr(objects[i], field)[0], getattr(objects[i], field)[1], color=colour)
            else:
                plt.scatter(getattr(objects[i], field)[0], getattr(objects[i], field)[1])

    def save_figure(self, name):
        """
        Save the figure with the name provided
        :param name: The name of the file to save the figure to
        """
        plt.savefig(f"{self.c.IMAGE_DIRECTORY}/{name}")
        plt.clf()
