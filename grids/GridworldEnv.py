import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, grid_size=4, max_steps=100):
        self.grid_size = grid_size
        self.max_steps = max_steps
        
        # Define movements
        self.movements = {0: [-1, 0],  # up
                          1: [1, 0],   # down
                          2: [0, -1],  # left
                          3: [0, 1],   # right
                          4: [0, 0]}   # stay
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.movements))
        self.observation_space = spaces.Discrete(self.grid_size**2)
        
        # Set rewards for special states
        self.rewarding_states = [(0, 0), (self.grid_size-1, self.grid_size-1)]
        self.rewarding_state_reward = 10
        self.normal_state_reward = -1
        
        self.seed()
        self.reset()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.current_step = 0
        self.agent_position = [self.np_random.randint(0, self.grid_size),
                               self.np_random.randint(0, self.grid_size)]
        return self._get_obs()
    
    def step(self, action):
        if action not in self.movements.keys():
            raise ValueError("Invalid action")
        
        # Calculate new position
        new_position = [sum(x) for x in zip(self.agent_position, self.movements[action])]
        
        # Check if new position is within the grid
        if (new_position[0] < 0) or (new_position[0] >= self.grid_size) or \
           (new_position[1] < 0) or (new_position[1] >= self.grid_size):
            new_position = self.agent_position
        
        # Update position
        self.agent_position = new_position
        
        # Get observation, reward, done and info
        obs = self._get_obs()
        done = (self.agent_position in self.rewarding_states) or (self.current_step >= self.max_steps)
        if self.agent_position in self.rewarding_states:
            reward = self.rewarding_state_reward
        else:
            reward = self.normal_state_reward
        
        info = {}
        
        # Increment step count and return step information
        self.current_step += 1
        return obs, reward, done, info
    
    def render(self, mode='human'):
        if mode == 'ansi':
            return self._to_string()
        elif mode == 'human':
            print(self._to_string())
        else:
            super(GridworldEnv, self).render(mode=mode)
            
    def _get_obs(self):
        return self.agent_position[0] * self.grid_size + self.agent_position[1]
    
    def _to_string(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[tuple(self.agent_position)] = 1
        for rew_state in self.rewarding_states:
            grid[tuple(rew_state)] = 10
        
        return str(grid)

