import torch
from unityagents import UnityEnvironment
import numpy as np


class Environment:
    def __init__(self, env_filepath):
        self.env = UnityEnvironment(file_name=env_filepath)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.reset()
        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.states.shape[1]
        self.agent_count = len(self.env_info.agents)

    def reset(self, train=True):
        self.env_info = self.env.reset(train_mode=train)[self.brain_name]

    def close(self):
        self.env.close()

    def step(self, actions):
        self.env_info = self.env.step(actions)[self.brain_name]
        next_obs = self.states
        rewards = np.array(self.env_info.rewards)
        dones = self.env_info.local_done
        return next_obs, rewards, dones

    @property
    def states(self):
        return torch.from_numpy(self.env_info.vector_observations).float()
