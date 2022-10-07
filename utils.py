import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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
    def __init__(self, lowdim_obs_shape, lowdim_qpos_shape, img_obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device
        self.lowdim_obses = np.empty((capacity, *lowdim_obs_shape), dtype=np.float32)
        self.lowdim_qpos = np.empty((capacity, *lowdim_qpos_shape), dtype=np.float32)
        self.next_lowdim_obses = np.empty((capacity, *lowdim_obs_shape), dtype=np.float32)
        self.next_lowdim_qpos = np.empty((capacity, *lowdim_qpos_shape), dtype=np.float32)
        self.hand_img_obses = np.empty((capacity, *img_obs_shape), dtype=np.uint8)
        self.next_hand_img_obses = np.empty((capacity, *img_obs_shape), dtype=np.uint8)
        self.third_person_img_obses = np.empty((capacity, *img_obs_shape), dtype=np.uint8)
        self.next_third_person_img_obses = np.empty((capacity, *img_obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.noisy_actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=bool)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=bool)

        self.idx = 0
        self.full = False

    def __len__(self):
        """ Moo Jin: Note that this definition is correct. There is no need to add +1 to self.idx. """
        return self.capacity if self.full else self.idx

    def add(self, lowdim_obs, lowdim_qpos, hand_img_obs, third_person_img_obs, action, naction, reward, next_lowdim_obs, next_lowdim_qpos, next_hand_img_obs, next_third_person_img_obs, done, done_no_max):
        np.copyto(self.lowdim_obses[self.idx], lowdim_obs)
        np.copyto(self.lowdim_qpos[self.idx], lowdim_qpos)
        np.copyto(self.hand_img_obses[self.idx], hand_img_obs)
        np.copyto(self.third_person_img_obses[self.idx], third_person_img_obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.noisy_actions[self.idx], naction)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_lowdim_obses[self.idx], next_lowdim_obs)
        np.copyto(self.next_lowdim_qpos[self.idx], next_lowdim_qpos)
        np.copyto(self.next_hand_img_obses[self.idx], next_hand_img_obs)
        np.copyto(self.next_third_person_img_obses[self.idx], next_third_person_img_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        lowdim_obs = self.lowdim_obses[idxs]
        next_lowdim_obs = self.next_lowdim_obses[idxs]

        hand_img_obses = self.hand_img_obses[idxs]
        next_hand_img_obses = self.next_hand_img_obses[idxs]
        third_person_img_obses = self.third_person_img_obses[idxs]
        next_third_person_img_obses = self.next_third_person_img_obses[idxs]

        lowdim_obs = torch.as_tensor(lowdim_obs, device=self.device).float()
        next_lowdim_obs = torch.as_tensor(next_lowdim_obs, device=self.device).float()
        hand_img_obses = torch.as_tensor(hand_img_obses, device=self.device).float()
        next_hand_img_obses = torch.as_tensor(next_hand_img_obses, device=self.device).float()
        third_person_img_obses = torch.as_tensor(third_person_img_obses, device=self.device).float()
        next_third_person_img_obses = torch.as_tensor(next_third_person_img_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return lowdim_obs, hand_img_obses, third_person_img_obses, actions, rewards, next_lowdim_obs, next_hand_img_obses, next_third_person_img_obses, not_dones_no_max