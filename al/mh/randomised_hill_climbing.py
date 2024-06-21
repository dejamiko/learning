from config import Config
from al.utils import NeighbourGenerator, set_seed
from al.mh.metaheuristic import MetaHeuristic
from playground.environment import Environment
from playground.object import TrajectoryObject


class RandomisedHillClimbing(MetaHeuristic):
    def __init__(self, c, similarity_dict, locked_subsolution, threshold=None):
        super().__init__(c, similarity_dict, locked_subsolution, threshold)

    def strategy(self):
        best_score = 0
        best_selection = None
        while self.count < self.c.MH_BUDGET:
            selected = self._get_random_initial_selection()
            curr_score = self.evaluate_selection(selected)
            while self.count < self.c.MH_BUDGET:
                next_selection = None
                next_score = 0
                for neighbour in NeighbourGenerator(selected, self.locked_subsolution):
                    if self.count >= self.c.MH_BUDGET:
                        break
                    score = self.evaluate_selection(neighbour)
                    if score > next_score:
                        next_selection = neighbour
                        next_score = score
                        if next_score > best_score:
                            best_selection = next_selection
                            best_score = next_score
                if next_score <= curr_score:
                    break
                selected = next_selection
                curr_score = next_score
        return best_selection


if __name__ == "__main__":
    config = Config()
    set_seed(config.SEED)
    env = Environment(config)
    similarity_dict = env.generate_objects_ail(TrajectoryObject)
    rhc = RandomisedHillClimbing(config, similarity_dict, [])

    selected = rhc.strategy()
    print(rhc.evaluate_selection(selected))
