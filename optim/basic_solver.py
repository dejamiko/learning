from config import Config
from optim.solver import Solver, evaluate_all_heuristics


class BasicSolver(Solver):
    def solve_one(self):
        selected = self.heuristic.solve()
        return self.env.evaluate_selection_transfer_based(selected)


if __name__ == "__main__":
    c = Config()

    results = evaluate_all_heuristics(BasicSolver, c, n=1)
    for name, mean, std, total_time in results:
        print(f"{name}: {mean} +/- {std}, time: {total_time}")
