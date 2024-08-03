from abc import ABC, abstractmethod

from optim.solver import Solver
from tm_utils import get_bin_representation


class ApproximationSolver(Solver, ABC):
    def __init__(self, config, heuristic_class):
        super().__init__(config, heuristic_class)

    def solve_one(self):
        selected = []
        while len(selected) < self.config.DEMONSTRATION_BUDGET:
            self.heuristic = self.heuristic_class(self.config, self.env, selected)
            heuristic_selected = self.heuristic.solve()
            obj_to_try = self._select_object_to_try(heuristic_selected)
            assert heuristic_selected[obj_to_try] == 1
            selected.append(obj_to_try)
            # update the lower and upper bounds based on the interactions of the selected object
            self._update_state(obj_to_try)
        count = self.env.evaluate_selection_transfer_based(
            get_bin_representation(selected, self.config.OBJ_NUM)
        )
        return count

    @abstractmethod
    def _select_object_to_try(self, selected):
        pass

    @abstractmethod
    def _update_state(self, obj_to_try):
        pass
