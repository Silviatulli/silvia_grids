import numpy as np
import gym

class GridworldEnv(gym.Env):
    def __init__(self):
        # Define movements
        self.movements = {0: [-1, 0],  # up
                          1: [1, 0],   # down
                          2: [0, -1],  # left
                          3: [0, 1],   # right
                          4: [0, 0]}   # stay

        # Define grid
        self.grid = np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]])

        # Define starting position of the agent
        self.starting_position = [2, 2]

        # Define rewarding and blocking states
        self.rewarding_states = [[0, 4]]
        self.blocking_states = []

        # Define current position of the agent
        self.current_position = self.starting_position

        # Define current step
        self.current_step = 0

        # Define maximum number of steps
        self.max_steps = 10

    def reset(self):
        # Reset current position
        self.current_position = self.starting_position

        # Reset current step
        self.current_step = 0

        # Return observation
        obs = np.array(self.grid)
        obs[tuple(self.current_position)] = 1
        return obs

    def step(self, action):
        # Increment current step
        self.current_step += 1

        # Get movement from action
        movement = self.movements[action]

        # Calculate new position
        new_position = np.add(self.current_position, movement)

        # Check if new position is valid
        if (new_position[0] < 0 or new_position[0] >= self.grid.shape[0] or
                new_position[1] < 0 or new_position[1] >= self.grid.shape[1] or
                list(new_position) in self.blocking_states):
            new_position = self.current_position

        # Update current position
        self.current_position = new_position

        # Get observation, reward, done and info
        obs = np.array(self.grid)
        obs[tuple(self.current_position)] = 1
        reward = 0
        done = (obs in self.rewarding_states) or (self.current_step >= self.max_steps)
        info = {}

        return obs, reward, done, info

    def render(self):
        # Print current position
        print(f"Current position: {self.current_position}")
