import numpy as np

from tm_utils import get_rng


class NeighbourGenerator:
    def __init__(self, selected, locked_subsolution, rng):
        self.selected = selected
        self.locked_subsolution = locked_subsolution
        self._rng = rng

    def __iter__(self):
        indices = self._rng.permutation(np.arange(len(self.selected)))
        indices2 = self._rng.permutation(np.arange(len(self.selected)))
        for i in indices:
            if self.selected[i] == 1 and i not in self.locked_subsolution:
                for j in indices2:
                    if self.selected[j] == 0:
                        new_selected = self.selected.copy()
                        new_selected[i] = 0
                        new_selected[j] = 1
                        yield new_selected, (i, j)


if __name__ == "__main__":
    selected = np.zeros(40)
    selected[0] = 1
    selected[1] = 1
    selected[2] = 1
    selected[3] = 1
    selected[4] = 1
    rng = get_rng(0)
    generator = NeighbourGenerator(selected, [], rng)
    print(len(list(generator)))
