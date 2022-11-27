from turtle import backward, forward
from robot_env import RobotEnv
import imageio
import math
import numpy as np
import os
import sys
import pickle as pkl
import gzip
from pathlib import Path
from utils import ReplayBuffer
from controllers import XboxController

class Workspace(object):
    def __init__(self, work_dir):
        self.work_dir = work_dir
        print(f'workspace: {self.work_dir}')
        self.hvideo_dir = self.work_dir / 'wrist_videos'
        self.hvideo_dir.mkdir(parents=True, exist_ok=True)
        self.tpvideo_dir = self.work_dir / 'third_person_videos'
        self.tpvideo_dir.mkdir(parents=True, exist_ok=True)

        # for saving forward trajectories
        self.forward_dir = self.work_dir / 'forward'
        self.forward_dir.mkdir(parents=True, exist_ok=True)
        self.forward_pkl = self.forward_dir / "replay_buffer.pkl"
        self.forward_demo_dir = self.forward_dir / "demos.npz"

        # for saving backward trajectories
        self.backward_dir = self.work_dir / 'backward'
        self.backward_dir.mkdir(parents=True, exist_ok=True)
        self.backward_pkl = self.backward_dir / "replay_buffer.pkl"
        self.backward_demo_dir = self.backward_dir / "demos.npz"

        # initialize robot environment
        self.DoF = 3
        self.env = RobotEnv(hz=10,
                            DoF=self.DoF,
                            ip_address='172.16.0.10',
                            randomize_ee_on_reset=True,
                            pause_after_reset=True,
                            hand_centric_view=True,
                            third_person_view=True,
                            qpos=True,
                            ee_pos=True,
                            local_cameras=False)
        self.max_length = 200
        self.controller = XboxController(DoF=self.DoF)

        continue_collection = False
        if os.path.exists(self.forward_pkl):
            user_input = input(f"A demo file already exists. Continue appending to it? (y) or (n): ")
            if user_input == 'y':
                continue_collection = True
                print('Continuing collection')
            else:
                print('Overwriting the original data')

        if continue_collection:
            with gzip.open(self.forward_pkl, "rb") as f:
                self.forward_buffer = pkl.load(f)
            with gzip.open(self.backward_pkl, "rb") as f:
                self.backward_buffer = pkl.load(f)
        else:
            self.forward_buffer = ReplayBuffer(self.env.observation_space,
                                               self.env.action_space,
                                               capacity=int(1e5),)
            self.backward_buffer = ReplayBuffer(self.env.observation_space,
                                                self.env.action_space,
                                                capacity=int(1e5),)
        self.verbose = True

    def momentum(self, delta, prev_delta):
        """Modifies action delta so that there is momentum (and thus less jerky movements)."""
        prev_delta = np.asarray(prev_delta)
        gamma = 0.15 # higher => more weight for past actions
        return (1 - gamma) * delta + gamma * prev_delta

    def record_gif(self, imgs, gif_log_path):
        """Saves a set of images as a gif (e.g., when you want a video of a demo)."""
        imageio.mimsave(gif_log_path, imgs, fps=self.env.hz)

    def single_demo(self, episode_num, reset=False):
        episode_reward, episode_step = 0, 0
        if reset:
            obs = self.env.reset()
        else:
            obs = self.env.get_observation()

        prev_action = np.zeros(self.DoF + 1)
        done = False

        current_episode = list()
        hand_imgs_for_gif = [obs['hand_img_obs']] # video of hand-centric demo
        third_person_imgs_for_gif = [obs['third_person_img_obs']] # video of third-person demo

        print(f'Starting episode {episode_num}...')
        while not done:
            print(f"episode_step: {episode_step}")

            # smoothen the action
            xbox_action = self.controller.get_action()
            smoothed_pos_delta = self.momentum(xbox_action[:self.DoF], prev_action[:self.DoF])
            action = np.append(smoothed_pos_delta, xbox_action[self.DoF]) # concatenate with gripper command

            next_obs, reward, done, _ = self.env.step(action)

            if self.verbose:
                print(f'commanded action: {xbox_action}')
                print(f'smoothed action: {action}')
                print(f"ee pos: {next_obs['lowdim_ee']}")

            # for the GIFs
            hand_imgs_for_gif.append(next_obs['hand_img_obs'])
            third_person_imgs_for_gif.append(next_obs['third_person_img_obs'])

            if episode_step == self.max_length - 1 or xbox_action[self.DoF + 1]:
                done = True

            episode_reward += reward
            current_episode.append((obs,
                                    action,
                                    reward,
                                    next_obs,
                                    done,))

            episode_step += 1
            prev_action = action
            obs = next_obs

        # save GIF of demos
        print('Saving videos (GIFs) of demo...')
        self.record_gif(hand_imgs_for_gif, os.path.join(self.hvideo_dir, f'{episode_num}.gif'))
        self.record_gif(third_person_imgs_for_gif, os.path.join(self.tpvideo_dir, f'{episode_num}.gif'))
        return current_episode

    def run(self):
        # assumes all episodes are of max length
        assert not self.forward_buffer.full, "The forward buffer is already full!"
        assert not self.backward_buffer.full, "The backward buffer is already full!"
        episode_num = 0
        reset_next_episode = True
        while (len(self.forward_buffer) + self.max_length <= self.forward_buffer.capacity) and \
              (len(self.backward_buffer) + self.max_length <= self.backward_buffer.capacity):
            current_episode = self.single_demo(episode_num, reset=reset_next_episode)
            user_input = input('Save current episode?: (f) for forward buffer, (b) for backward buffer and (d) to discard episode and (q) to save and quit: ')
            if user_input.startswith('f'):
                for transition in current_episode:
                    self.forward_buffer.add(*transition)
                print("Added to the forward demos. Current number of steps in the buffer:", len(self.forward_buffer))
            if user_input.startswith('b'):
                for transition in current_episode:
                    self.backward_buffer.add(*transition)
                print("Added to the backward demos. Current number of steps in the buffer:", len(self.backward_buffer))
            if user_input.startswith('d'): # discard episode
                pass
            if user_input.startswith('q'): # save episode and quit
                break
            episode_num += 1
            reset_next_episode = user_input.endswith('j')

        print('Saving demos and exporting buffer to pickle file...')
        self.forward_buffer.save(self.forward_demo_dir)
        with gzip.open(self.forward_pkl, 'wb') as f:
            pkl.dump(self.forward_buffer, f, protocol=pkl.HIGHEST_PROTOCOL)
        self.backward_buffer.save(self.backward_demo_dir)
        with gzip.open(self.backward_pkl, 'wb') as f:
            pkl.dump(self.backward_buffer, f, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    base_dir = Path('/home/panda5/franka_demos/franka_demos')
    work_dir = base_dir / sys.argv[1]
    workspace = Workspace(work_dir=work_dir)
    workspace.run()