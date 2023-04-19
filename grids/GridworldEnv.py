import gym
import numpy as np

import gym
import numpy as np
from gym import spaces
from typing import Tuple

class CustomGridworldEnv(gym.Env):
    
    def __init__(self):
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(5)  # Five discrete actions: 0 for up, 1 for down, 2 for left, 3 for right, 4 for stay
        self.observation_space = gym.spaces.Discrete(19)  # 19 discrete observations representing the 19 cells in the gridworld

        # Define gridworld properties
        self.grid = np.array([[1, 2, 3, 4, 5],
                              [6, 0, 7, 0, 8],
                              [9, 0, 10, 0, 11],
                              [12, 0, 13, 0, 14],
                              [15, 16, 17, 18, 19]])
        self.agent_position = [4, 4]  # Agent starts at bottom-right corner
        self.rewarding_states = [3, 9, 10, 11]
        self.reward_values = [1, -1, -1, -1]  # Positive reward for state 3, Negative reward for states 9, 10, and 11; 
        self.current_step = 0  # Current step counter
        self.max_steps = 100  # Maximum number of steps per episode

    def reset(self) -> int:
        self.agent_position = [4, 4]
        self.current_step = 0
        return self.grid[tuple(self.agent_position)]

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        # Define movements
        movements = {0: [-1, 0],  # up
                     1: [1, 0],   # down
                     2: [0, -1],  # left
                     3: [0, 1],   # right
                     4: [0, 0]}   # stay
        
        # Get the new position of the agent
        new_position = [sum(x) for x in zip(self.agent_position, movements[action])]
        
        # Check if the new position is valid, if not, stay in the same position
        if new_position[0] < 0 or new_position[0] > 4 or new_position[1] < 0 or new_position[1] > 4:
            new_position = self.agent_position
        
        # Update agent position
        self.agent_position = new_position
        
        # Get observation, reward, done and info
        obs = self.grid[tuple(self.agent_position)]
        done = (obs in self.rewarding_states) or (self.current_step >= self.max_steps)
        reward = self.reward_values[self.rewarding_states.index(obs)] if done else 0
        info = {}
        
        # Update step counter
        self.current_step += 1
        
        return obs, reward, done, info

