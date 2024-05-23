import numpy as np


class Environment:
    def __init__(self):
        self.goal = None
        self.generate_random_goal()
        self.agent_position = np.zeros(3)
        self.current_step = 0
        self.max_steps = 5000
        self.max_action = 0.1

    def generate_random_goal(self):
        self.goal = np.random.uniform(-1, 1, 3)

    def get_reward(self):
        # for now dense reward
        return -np.linalg.norm(self.agent_position - self.goal)

    def get_state(self):
        return self.agent_position

    def step(self, action):
        self.current_step += 1
        if self.current_step >= self.max_steps:
            return self.get_state(), -100, True, True, self.current_step

        action = np.clip(action, -self.max_action, self.max_action)
        self.agent_position += action
        if np.linalg.norm(self.agent_position - self.goal) < 0.1:
            return self.get_state(), 100, True, False, self.current_step
        return self.get_state(), self.get_reward(), False, False, None

    def reset(self):
        self.agent_position = np.zeros(3)
        self.generate_random_goal()
        self.current_step = 0
        return self.get_state(), None
