import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, observation_space, action_space, capacity):
        self.capacity = capacity
        self.observations = {}
        self.next_observations = {}
        for key, obs_space in observation_space.items():
            self.observations[key] = np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
            self.next_observations[key] = np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
        self.actions = np.empty((capacity, *action_space.shape), dtype=action_space.dtype)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=bool)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self,
            observation,
            action,
            reward,
            next_observation,
            done,):

        for key in self.observations.keys():
            np.copyto(self.observations[key][self.idx], observation[key])
        for key in self.next_observations.keys():
            np.copyto(self.next_observations[key][self.idx], next_observation[key])
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.dones[self.idx], done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def save(self, directory):
        total_steps = len(self)
        save_dict = {}
        for key, val in self.observations.items():
            save_dict[key] = val[:total_steps]
        for key, val in self.next_observations.items():
            save_dict[f'next_{key}'] = val[:total_steps]
        save_dict['actions'] = self.actions[:total_steps]
        save_dict['rewards'] = self.rewards[:total_steps]
        save_dict['dones'] = self.dones[:total_steps].astype(int)
        np.savez(directory, **save_dict)