import numpy as np


class NeighbourGenerator:
    def __init__(self, selected, locked_subsolution):
        self.selected = selected
        self.locked_subsolution = locked_subsolution

    def __iter__(self):
        indices = np.random.permutation(np.arange(len(self.selected)))
        indices2 = np.random.permutation(np.arange(len(self.selected)))
        for i in indices:
            if self.selected[i] == 1 and i not in self.locked_subsolution:
                for j in indices2:
                    if self.selected[j] == 0:
                        new_selected = self.selected.copy()
                        new_selected[i] = 0
                        new_selected[j] = 1
                        yield new_selected
