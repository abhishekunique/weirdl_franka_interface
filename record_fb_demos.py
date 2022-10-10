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

        load_demo = False
        self.f_demo_file = f"{self.work_dir}/f_demos.pkl"
        self.b_demo_file = f"{self.work_dir}/b_demos.pkl"

        # Check if demo file exists already; if so, warn about overwrite (unless user is intentionally loading an existing one).
        if cfg.load_demo:
            self.f_replay_buffer = pkl.load(open(self.f_demo_file, "rb")) #load previous collection attempt (allows you to collect in multiple sessions)
            self.b_replay_buffer = pkl.load(open(self.b_demo_file, "rb"))
            print(f'\n forward buffer has {len(self.f_replay_buffer)} steps \n')
            print(f'\n backward buffer has {len(self.b_replay_buffer)} steps \n')
            return
        else:
            if os.path.isfile(self.f_demo_file) and os.path.isfile(self.b_demo_file):
                user_input = input(f"A foward demo file ({self.f_demo_file}) already exists. If you continue, it will be overwritten alongside the backward demos ({self.b_demo_file}). (To load an existing replay buffer and continue adding to it, set load_demo to `true` in oculus_demo.yaml.) Continue? (y) or (n): ")
                if user_input != 'y':
                    exit(0)
            
            self.f_replay_buffer = ReplayBuffer(self.env.lowdim_space.shape,
                                              self.env.qpos_space.shape,
                                              self.env.observation_space['hand_img_obs'].shape,
                                              self.env.action_space.shape,
                                              25000,
                                              self.device)
            self.b_replay_buffer = ReplayBuffer(self.env.lowdim_space.shape,
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

    def single_b_demo(self, episode_num, cfg):
        episode, episode_reward, episode_step, done = 1, 0, 0, True
        action_delta_limit = 1.0
        if cfg.backward:
            # Manually send to rest and get
            # obs so we can attach objects 
            # to gripper
            print(f'Starting backward episode {episode_num}...')
            user_input = input("Enter (s) to wait 5 seconds & anything else to continue: ")
            if user_input in ['s', 'S']:
                time.sleep(5)
            else:
                # otherwise just continue
                pass
            obs = self.env.get_observation()
        else:
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
                print(f"\t gripper action { action[3]}")
                print(f"gripper flag {self.env.curr_gripper_position}")
                print(f' gripper state {lowdim_qpos[7]}')
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
        self.record_gif(hand_imgs_for_gif, f'{self.work_dir}/b_hand_video.gif')
        self.record_gif(third_person_imgs_for_gif, f'{self.work_dir}/b_third_person_video.gif')

        user_input = input("Enter (s) to save this demo, (d) to discard, (j) to discard and return to rest (k) to discard and reset joints: ")
        return user_input, buffer_list

    def single_f_demo(self, episode_num, cfg):
        episode, episode_reward, episode_step, done = 1, 0, 0, True
        action_delta_limit = 1.0
        if cfg.backward:
            # Manually send to rest and get
            # obs so we can attach objects 
            # to gripper
            print(f'Starting forward episode {episode_num}...')
            user_input = input("Enter (s) to wait 5 seconds & anything else to continue: ")
            if user_input in ['s', 'S']:
                time.sleep(5)
            else:
                # otherwise just continue
                pass
            obs = self.env.get_observation()
        else:
            obs = self.env.reset()
        lowdim_qpos, lowdim_obs, hand_img_obs, third_person_img_obs = obs['lowdim_qpos'], obs['lowdim_obs'], obs['hand_img_obs'], obs['third_person_img_obs']
        done = False
        success = 0
        buffer_list = list()
        prev_modified_delta = np.zeros(3)

        hand_imgs_for_gif = [] # video of hand-centric demo
        third_person_imgs_for_gif = [] # video of third-person demo

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
            if cfg.verbose:
                print(f'action fronm xbox {action}')
                print(f'noisy action {noisy_action}')
                print(f'commanded action {action}')
                print(f'qpos {list(lowdim_qpos)}')
                print(f'qpos shape {lowdim_qpos.shape}')
                print(f"gripper flag {self.env.curr_gripper_position}")
                print(f"\t gripper action { action[3]}")
                print(f' gripper state {lowdim_qpos[7]}')
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
        self.record_gif(hand_imgs_for_gif, f'{self.work_dir}/f_hand_video.gif')
        self.record_gif(third_person_imgs_for_gif, f'{self.work_dir}/f_third_person_video.gif')

        user_input = input(" (s) save this demo, (j) save & reset joints (r) only to save backward (e) to try again (d) to discard both demos, (k) to discard and reset joints (q) save and quit: ")
        # If trying again set to None to remain in loop
        user_input = None if user_input in ['e'] else user_input
        return user_input, buffer_list


    def run(self, cfg):
        b_episode_num = f_episode_num = 1
        assert not self.f_replay_buffer.full, "The forward replay buffer is already full!"
        f_user_input = b_user_input = None
        num_demos_remaining =  math.floor((self.f_replay_buffer.capacity - len(self.f_replay_buffer)) / self.env.max_path_length) # i.e., floor[(capacity_in_steps - num_steps_currently_in_buffer) / (max_num_steps_per_episode)]
        if cfg.backward:
                # call reset once to avoid error
                _ = self.env.reset()
        while num_demos_remaining > 0:
            f_user_input = None
            # collect backward, wait for user input
            # if successful continue to forward
            # & save demo. Otherwise discard
            b_user_input, buffer_list_b = self.single_b_demo(b_episode_num)
            self.step = 0
            if b_user_input == 'd': # discard episode
                pass
            elif b_user_input == 'j': # discard and return EE to center
                self.env.reset()

            # Collect forward only if backward was successful
            if b_user_input != 's':
                continue
            # if user returns (e), try again
            while f_user_input is None:
                f_user_input, buffer_list_f = self.single_f_demo(f_episode_num)
                self.step = 0

            grip_input = input("To open gripper enter (o): ")
            # potentially open gripper to reset env for backward
            # demos
            if grip_input == 'o':
                self.env.reset_gripper()
            elif f_user_input == 'd': # discard episode
                # if we discard forward, neither
                # demos will be saved and we try again
                pass
           
            if b_user_input in ['s'] and f_user_input in ['s', 'j', 'q', 'r']:
                b_episode_num += 1
                if f_user_input != 'r':
                    f_episode_num += 1
                # add demos to both buffers
                if f_user_input != 'r':
                    for buf in buffer_list_f:
                        self.f_replay_buffer.add(*buf) # Add demo steps to replay buffer
                for buf in buffer_list_b:
                    self.b_replay_buffer.add(*buf)
                print(f"Added 1 demo to both buffers. backward buffer has {len(self.b_replay_buffer)} forward has {len(self.f_replay_buffer)}")
            if f_user_input == 'q': # save both episodes and quit
                break

        print('Exporting forward demos to pickle file...')
        num_total_steps_f = len(self.f_replay_buffer)
        fp_imgs_f = self.f_replay_buffer.hand_img_obses[:num_total_steps_f]
        tp_imgs_f = self.f_replay_buffer.third_person_img_obses[:num_total_steps_f]
        imgs_f = np.concatenate([fp_imgs_f, tp_imgs_f], axis=1)
        state_f = self.f_replay_buffer.lowdim_obses[:num_total_steps_f]
        actions_f = self.f_replay_buffer.actions[:num_total_steps_f]
        qpos_f = self.f_replay_buffer.lowdim_qpos[:num_total_steps_f]
        noisy_actions_f = self.f_replay_buffer.noisy_actions[:num_total_steps_f]
        print((actions_f == noisy_actions_f).all())
        # terminals are weird, manually overwrite
        # since we don't call reset
        term = np.zeros((100, 1))
        term[99] = 1
        # ensure state & actions are normalized
        if float(np.amax(actions_f)) > 1.0 or float(np.amax(state_f)) != 1.0 or float(np.amin(actions_f)) < -1.0 or float(np.amin(state_f)) != -1.0:
            print(f'Forward action / state bounds violated!')
            print(f' state max {np.amax(state_f)}')
            print(f'state min {np.amin(state_f)}')
        # rewards are always zero
        rewards_f = np.zeros((len(self.f_replay_buffer), 1))
        done_bool_f = ~self.f_replay_buffer.not_dones[:num_total_steps_f]
        # ARL saves terminals as int
        terminals_f = done_bool_f.astype(int)
        terminals_f = np.concatenate([term for i in range(int(num_total_steps_f/100))], axis=0)
        np.savez(f'{self.work_dir}/forward_redcube.npz', state=state_f, imgs=imgs_f, qpos=qpos_f, noisy_actions=noisy_actions_f, actions=actions_f, rewards=rewards_f, terminals=terminals_f, allow_pickle=True)
        pkl.dump(self.f_replay_buffer, open(cfg.f_demo_file, "wb"), protocol=pkl.HIGHEST_PROTOCOL)

        print('Exporting backward demos to pickle file...')
        num_total_steps_b = len(self.b_replay_buffer)
        fp_imgs_b = self.b_replay_buffer.hand_img_obses[:num_total_steps_b]
        tp_imgs_b = self.b_replay_buffer.third_person_img_obses[:num_total_steps_b]
        imgs_b= np.concatenate([fp_imgs_b, tp_imgs_b], axis=1)
        state_b = self.b_replay_buffer.lowdim_obses[:num_total_steps_b]
        actions_b = self.b_replay_buffer.actions[:num_total_steps_b]
        qpos_b = self.b_replay_buffer.lowdim_qpos[:num_total_steps_b]
        noisy_actions_b = self.b_replay_buffer.noisy_actions[:num_total_steps_b]
        print((actions_b == noisy_actions_b).all())
        
        # ensure state & actions are normalized
        if float(np.amax(actions_b)) > 1.0 or float(np.amax(state_b)) != 1.0 or float(np.amin(actions_b)) < -1.0 or float(np.amin(state_b)) != -1.0:
            print(f'Backward action / state bounds violated!')
            print(f' state max {np.amax(state_b)}')
            print(f'state min {np.amin(state_b)}')
        # rewards are always zero
        rewards_b = np.zeros((len(self.b_replay_buffer), 1))
        done_bool_b = ~self.b_replay_buffer.not_dones[:num_total_steps_b]
        # ARL saves terminals as int
        terminals_b = done_bool_b.astype(int)
        terminals_b = np.concatenate([term for i in range(int(num_total_steps_b/100))], axis=0)
        np.savez(f'{self.work_dir}/backward_redcube.npz', state=state_b, imgs=imgs_b, qpos=qpos_b, noisy_actions=noisy_actions_b, actions=actions_b, rewards=rewards_b, terminals=terminals_b, allow_pickle=True)
        pkl.dump(self.b_replay_buffer, open(cfg.b_demo_file, "wb"), protocol=pkl.HIGHEST_PROTOCOL)


@hydra.main(config_path='../config/', config_name='fb_demo')
def main(cfg):
    # from record_oculus_demos import Workspace as W
    workspace = Workspace(cfg)
    workspace.run(cfg)
        
if __name__ == '__main__':
    main()