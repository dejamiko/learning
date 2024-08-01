from config import Config
from optim.mh.metaheuristic import MetaHeuristic
from optim.utils import NeighbourGenerator
from playground.environment import Environment


class RandomisedHillClimbing(MetaHeuristic):
    def __init__(self, c, similarity_dict, locked_subsolution):
        super().__init__(c, similarity_dict, locked_subsolution)
        self.best_selection = None

    def strategy(self):
        best_score = 0
        while self.count < self.c.MH_BUDGET:
            selected = self.get_random_selection()
            curr_score = self.evaluate_selection(selected)
            while self.count < self.c.MH_BUDGET:
                next_selection = None
                next_score = 0
                for neighbour in NeighbourGenerator(
                    selected, self.locked_subsolution, self._rng
                ):
                    if self.count >= self.c.MH_BUDGET:
                        break
                    score = self.evaluate_selection(neighbour)
                    if score > next_score:
                        next_selection = neighbour
                        next_score = score
                        if next_score > best_score:
                            self.best_selection = next_selection
                            best_score = next_score
                if next_score <= curr_score:
                    break
                selected = next_selection
                curr_score = next_score
        return self.best_selection

    def get_best_solution(self):
        return self.best_selection


if __name__ == "__main__":
    config = Config()
    env = Environment(config)
    rhc = RandomisedHillClimbing(config, env, [])

    selected = rhc.strategy()
    print(rhc.evaluate_selection(selected))
