from config import Config
from optim.approx_solver import ApproximationSolver
from optim.solver import evaluate_all_heuristics


class LinearApproximationSolver(ApproximationSolver):
    # TODO finish this implementation
    def __init__(self, config, heuristic_class):
        # it only makes sense to use the real-valued version here
        config.SUCCESS_RATE_BOOLEAN = False
        super().__init__(config, heuristic_class)
        self.lin_fun = (1, 0)  # 1 * x + 0

    def _select_object_to_try(self, selected):
        pass

    def _update_state(self, obj_to_try):
        self.env.update_visual_similarities(self.lin_fun)

    def _init_data(self, i):
        super()._init_data(i)
        self._reset_function()

    def _reset_function(self):
        self.lin_fun = (1, 0)


if __name__ == "__main__":
    c = Config()

    results = evaluate_all_heuristics(LinearApproximationSolver, c, n=1)
    for name, mean, std, total_time in results:
        print(f"{name}: {mean} +/- {std}, time: {total_time}")
