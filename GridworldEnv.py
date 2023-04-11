import gym
from gym import spaces
import numpy as np

class GridworldEnv(gym.Env):
    def __init__(self):
        super(GridworldEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(5)  # Five discrete actions: 0 for up, 1 for down, 2 for left, 3 for right, 4 for stay
        self.observation_space = spaces.Discrete(19)  # 19 discrete observations representing the 19 cells in the gridworld

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

    def reset(self):
      # Reset environment and return initial observation
      self.agent_position = [4, 4]
      self.current_step = 0
      return self._get_observation(), {}  # Return tuple of obs and reset infonfo

    def step(self, action):
        # Update environment based on given action and return observation, reward, done, and info
        self.current_step += 1

        # Update agent position based on action
        if action == 0:  # Move up
            self.agent_position[0] = max(0, self.agent_position[0] - 1)
        elif action == 1:  # Move down
            self.agent_position[0] = min(4, self.agent_position[0] + 1)
        elif action == 2:  # Move left
            self.agent_position[1] = max(0, self.agent_position[1] - 1)
        elif action == 3:  # Move right
            self.agent_position[1] = min(4, self.agent_position[1] + 1)
        else:  # Stay
            pass

        # Get current observation
        observation = self._get_observation()

        # Compute reward based on current
        reward = self.reward_values[self.grid[self.agent_position[0], self.agent_position[1]] - 1]

        # Check if the agent has reached a terminal state
        done = False
        truncated = False

        if self.grid[self.agent_position[0], self.agent_position[1]] in self.rewarding_states:
            done = True

        # Check if maximum number of steps reached
        if self.current_step >= self.max_steps:
            done = True
            truncated = True

        # Set info as an empty dictionary
        info = {}


        return observation, reward, done, truncated, info

    def _get_observation(self):
        # Get current observation based on agent position
        return self.grid[self.agent_position[0], self.agent_position[1]]

    def render(self, mode='human'):
        # Render gridworld state
        if mode == 'human':
            for row in self.grid:
                print(' ----' * 5 + ' ')
                print('|', end=' ')
                for col in row:
                    if col == 0:
                        print('   |', end=' ')
                    else:
                        print(f' {col} |', end=' ')
                print('')
            print(' ----' * 5 + ' ')
        elif mode == 'ansi':
            pass  # TODO: Implement ASCII rendering for other modes