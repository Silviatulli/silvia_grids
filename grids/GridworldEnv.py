import gym
import numpy as np

class CustomGridworld(gym.Env):
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

    def _get_obs(self):
        # Flatten the gridworld array
        flattened_grid = self.grid.flatten()
        
        # Get the index of the agent's current cell in the flattened array
        agent_index = np.ravel_multi_index(self.agent_position, self.grid.shape)
        
        # Create the observation array by setting the agent's current cell to 1 and all other cells to 0
        obs = np.zeros_like(flattened_grid)
        obs[agent_index] = 1
        
        return obs

    def reset(self):
        self.agent_position = [4, 4]
        self.current_step = 0
        obs = self._get_obs()
        return obs

    def step(self, action):
        # Take action and move agent
        if action == 0:  # Up
            self.agent_position[0] = max(self.agent_position[0] - 1, 0)
        elif action == 1:  # Down
            self.agent_position[0] = min(self.agent_position[0] + 1, self.grid.shape[0] - 1)
        elif action == 2:  # Left
            self.agent_position[1] = max(self.agent_position[1] - 1, 0)
        elif action == 3:  # Right
            self.agent_position[1] = min(self.agent_position[1] + 1, self.grid.shape[1] - 1)

        # Get observation and reward
        obs = self._get_obs()
        reward = 0
        if self.agent_position in self.rewarding_states:
            reward = self.reward_values[self.rewarding_states.index(self.agent_position)]

        # Check if episode is done
        done = False
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return obs, reward, done, {}

