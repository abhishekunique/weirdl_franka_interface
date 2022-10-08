import copy
import csv
from .robot_env import RobotEnv
import gym
import imageio
import importlib
import math
import numpy as np
import os
import pickle as pkl
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from shutil import copyfile
from iris_robots.utils import ReplayBuffer, set_seed_everywhere
from iris_robots.controllers import XboxController

torch.backends.cudnn.benchmark = True

class Workspace(object):
    def __init__(self):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        print("cuda status: ", torch.cuda.is_available())
        
        set_seed_everywhere(0)
        self.device = torch.device('cuda')
        self.verbose = False
        # init robot env
        env_kwargs = {
        'ip_address' : '172.24.68.68',
        'use_hand_centric_view': True,
        'use_third_person_view': True,
        'random_reset': True,
        'hz': 10,
        'pause_resets': True,
        'viz' : False,
        'demo_collection_mode': True,
        }
        self.env = RobotEnv(**env_kwargs)
        self.step = 0
        self.controller = XboxController(DoF=3)
        self.trigger_start_time = time.time()

        load_demo = True
        self.demo_file = f"{self.work_dir}/demos.pkl"

        # Check if demo file exists already; if so, warn about overwrite (unless user is intentionally loading an existing one).
        if load_demo:
            self.replay_buffer = pkl.load(open(self.demo_file, "rb")) # load previous collection attempt (allows you to collect in multiple sessions)
            return
        else:
            if os.path.isfile(self.demo_file):
                user_input = input(f"A demo file already exists. If you continue, it will be overwritten. Continue? (y) or (n): ")
                if user_input != 'y':
                    exit(0)
            self.replay_buffer = ReplayBuffer(self.env.lowdim_space.shape,
                                              self.env.qpos_space.shape,
                                              self.env.observation_space['hand_img_obs'].shape,
                                              self.env.action_space.shape,
                                              25000,
                                              self.device)

    def transform_basis(self, delta):
        """Change axes of action deltas to match an intuitive front-facing control of the robot."""
        delta[0], delta[1], delta[2] = -delta[2], -delta[0], delta[1]
        return delta

    def momentum(self, delta, prev_delta):
        """Modifies action delta so that there is momentum (and thus less jerky movements)."""
        prev_delta = np.asarray(prev_delta)
        gamma = 0.15 # the higher this is, the more that past actions are factored in and the less effect current action has
        return (1 - gamma) * delta + gamma * prev_delta

    def record_gif(self, imgs, gif_log_path):
        """Saves a set of images as a gif (e.g., when you want a video of a demo)."""
        imgs = [np.transpose(img, (1, 2, 0)) for img in imgs] # e.g. (3, 100, 100) -> (100, 100, 3) bc the latter is needed to save gif
        imageio.mimsave(gif_log_path, imgs, fps=self.env.hz)

    def single_demo(self, episode_num):
        episode, episode_reward, episode_step, done = 1, 0, 0, True
        action_delta_limit = 1.0
        obs = self.env.reset()
        lowdim_qpos, lowdim_obs, hand_img_obs, third_person_img_obs = obs['lowdim_qpos'], obs['lowdim_obs'], obs['hand_img_obs'], obs['third_person_img_obs']
        done = False
        success = 0
        buffer_list = list()
        prev_modified_delta = np.zeros(3)

        hand_imgs_for_gif = [] # video of hand-centric demo
        third_person_imgs_for_gif = [] # video of third-person demo

        print(f'Starting episode {episode_num}...')
        while self.step < self.env.max_path_length:
            print("\tstep:", self.step) # the step variable increments until the end of episode
            
            # Record GIFs:
            hand_imgs_for_gif.append(hand_img_obs)
            third_person_imgs_for_gif.append(third_person_img_obs)

            # Add momentum to action delta to make motions smoother
            action = self.controller.get_action()
            modified_delta = action[:3]
            modified_delta = self.momentum(modified_delta, prev_modified_delta)
           
            action = np.append(modified_delta, action[3]) # concatenate with gripper command

            # concat gripper info again
            noisy_action = modified_delta + 0*np.random.uniform(-0.05, 0.05, (3,))
            # clip after adding noise
            noisy_action = np.clip(noisy_action, -1, 1)
            # concat gripper info again
            noisy_action = np.append(noisy_action, action[3])
            if self.verbose:
                print(f'action fronm xbox {action}')
                print(f'noisy action {noisy_action}')
                print(f'commanded action {action}')
                print(f'qpos {list(lowdim_qpos)}')
                print(f'qpos shape {lowdim_qpos.shape}')
            obs, reward, done, info = self.env.step(noisy_action)
            next_lowdim_qpos, next_lowdim_obs, next_hand_img_obs, next_third_person_img_obs = obs['lowdim_qpos'], obs['lowdim_obs'], obs['hand_img_obs'], obs['third_person_img_obs']

            # allow infinite bootstrap
            done = bool(done)
            # no max is not used
            done_no_max = done
            episode_reward += reward

            buffer_list.append((lowdim_obs,
                                lowdim_qpos,
                                hand_img_obs,
                                third_person_img_obs,
                                action,
                                noisy_action,
                                reward,
                                next_lowdim_obs,
                                next_lowdim_qpos,
                                next_hand_img_obs,
                                next_third_person_img_obs,
                                done,
                                done_no_max))

            hand_img_obs = next_hand_img_obs
            third_person_img_obs = next_third_person_img_obs
            lowdim_obs = next_lowdim_obs
            lowdim_qpos = next_lowdim_qpos
            episode_step += 1
            self.step += 1
            prev_modified_delta = modified_delta

        # Save GIF of demo so that user can see whether it's good or not before saving/discarding.
        print('Saving videos (GIFs) of demo...')
        self.record_gif(hand_imgs_for_gif, f'{self.work_dir}/hand_video.gif')
        self.record_gif(third_person_imgs_for_gif, f'{self.work_dir}/third_person_video.gif')

        user_input = input("Enter (s) to save this demo, (d) to discard (q) to save and quit: ")
        if user_input in ['s', 'j', 'q']:
            for buf in buffer_list:
                self.replay_buffer.add(*buf) # Add demo steps to replay buffer
            print("Added 1 demo to the replay buffer. Current number of steps in the buffer:", len(self.replay_buffer))
        else:
            print("Discarded the demo!")
        return user_input


    def run(self):
        episode_num = 1
        assert not self.replay_buffer.full, "The replay buffer is already full!"
        num_demos_remaining =  math.floor((self.replay_buffer.capacity - len(self.replay_buffer)) / self.env.max_path_length) # i.e., floor[(capacity_in_steps - num_steps_currently_in_buffer) / (max_num_steps_per_episode)]
        while num_demos_remaining > 0:
            user_input = self.single_demo(episode_num)
            self.step = 0
            if user_input == 's': # save episode
                episode_num += 1
            if user_input == 'd': # discard episode
                pass
            if user_input == 'k': # discard episode and reset joints
                self.env.reset_joints()
            if user_input == 'q': # save episode and quit
                break

        print('Exporting demos to pickle file...')
        num_total_steps = len(self.replay_buffer)
        fp_imgs = self.replay_buffer.hand_img_obses[:num_total_steps]
        tp_imgs = self.replay_buffer.third_person_img_obses[:num_total_steps]
        imgs = np.concatenate([fp_imgs, tp_imgs], axis=1)
        state = self.replay_buffer.lowdim_obses[:num_total_steps]
        actions = self.replay_buffer.actions[:num_total_steps]
        qpos = self.replay_buffer.lowdim_qpos[:num_total_steps]
        noisy_actions = self.replay_buffer.noisy_actions[:num_total_steps]
        print((actions == noisy_actions).all())
        # terminals are weird, manually overwrite
        # ensure state & actions are normalized
        if float(np.amax(actions)) > 1.0 or float(np.amax(state)) != 1.0 or float(np.amin(actions)) < -1.0 or float(np.amin(state)) != -1.0:
            print(f'action / state bounds violated!')
            print(f' state max {np.amax(state)}')
            print(f'state min {np.amin(state)}')
        # rewards are always zero
        rewards = np.zeros((len(self.replay_buffer), 1))
        done_bool = ~self.replay_buffer.not_dones[:num_total_steps]
        # ARL saves terminals as int
        terminals = done_bool.astype(int)
        np.savez(f'redcube.npz', state=state, imgs=imgs, qpos=qpos, noisy_actions=noisy_actions, actions=actions, rewards=rewards, terminals=terminals, allow_pickle=True)
        pkl.dump(self.replay_buffer, open(self.demo_file, "wb"), protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    workspace = Workspace()
    workspace.run()