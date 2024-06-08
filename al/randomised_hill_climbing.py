from config import Config
from utils import MetaHeuristic, NeighbourGenerator


class RandomisedHillClimbing(MetaHeuristic):
    def __init__(self, c):
        super().__init__(c)
        self.best_selection = None

    def strategy(self):
        best_score = 0
        for i in range(self.c.RHC_ITER):
            selected = self._get_initial_selection()
            curr_score = self.evaluate_selection(selected)
            count = 0
            while True:
                count += 1
                next_selection = None
                next_score = 0
                for neighbour in NeighbourGenerator(selected):
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


if __name__ == "__main__":
    c = Config()

    randomised_hill_climbing = RandomisedHillClimbing(c)
    print(f"Randomised hill climbing selection for {c.RHC_ITER} iterations")
    mean, std = randomised_hill_climbing.evaluate_strategy(n=5)
    print(f"Mean: {mean}, std: {std}, time taken: {randomised_hill_climbing.get_mean_time()}")
