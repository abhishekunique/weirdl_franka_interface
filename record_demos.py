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
        self.tp_failed_video_dir = self.work_dir / 'failed_demos' / 'third_person_videos'
        self.tp_failed_video_dir.mkdir(parents=True, exist_ok=True)
        self.fp_failed_video_dir = self.work_dir / 'failed_demos' / 'wrist_videos'
        self.fp_failed_video_dir.mkdir(parents=True, exist_ok=True)
        self.rb_pkl = self.work_dir / "replay_buffer.pkl"
        self.demo_dir = self.work_dir / "demos.npz"

        # initialize robot environment
        self.env = RobotEnv(hz=10,
                            ip_address='172.16.0.10',
                            randomize_ee_on_reset=True,
                            pause_after_reset=True,
                            hand_centric_view=True,
                            third_person_view=True,
                            qpos=True,
                            ee_pos=True,
                            local_cameras=False)
        self.max_length = 200
        self.controller = XboxController(DoF=3)

        continue_collection = False
        if os.path.exists(self.rb_pkl):
            user_input = input(f"A demo file already exists. Continue appending to it? (y) or (n): ")
            if user_input == 'y':
                continue_collection = True
                print('Continuing collection')
            else:
                print('Overwriting the original data')

        if continue_collection:
            with gzip.open(self.rb_pkl, "rb") as f:
                self.replay_buffer = pkl.load(f)
        else:
            self.replay_buffer = ReplayBuffer(self.env.observation_space,
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
        # imgs = [np.transpose(img, (1, 2, 0)) for img in imgs] # e.g. (3, 100, 100) -> (100, 100, 3)
        imageio.mimsave(gif_log_path, imgs, fps=self.env.hz)

    def single_demo(self, episode_num):
        episode_reward, episode_step = 0, 0
        obs = self.env.reset()
        prev_action = np.zeros(4)
        done = False

        current_episode = list()
        hand_imgs_for_gif = [obs['hand_img_obs']] # video of hand-centric demo
        third_person_imgs_for_gif = [obs['third_person_img_obs']] # video of third-person demo

        print(f'Starting episode {episode_num}...')
        while not done:
            print(f"episode_step: {episode_step}")

            # smoothen the action
            xbox_action = self.controller.get_action()
            smoothed_pos_delta = self.momentum(xbox_action[:3], prev_action[:3])
            action = np.append(smoothed_pos_delta, xbox_action[3]) # concatenate with gripper command

            next_obs, reward, done, _ = self.env.step(action)

            if self.verbose:
                print(f'commanded action: {xbox_action}')
                print(f'smoothed action: {action}')
                print(f"ee pos: {next_obs['lowdim_ee']}")

            # for the GIFs
            hand_imgs_for_gif.append(next_obs['hand_img_obs'])
            third_person_imgs_for_gif.append(next_obs['third_person_img_obs'])

            if episode_step == self.max_length - 1:
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

        user_input = input("Enter (s) to save this demo, (d) to discard (q) to save and quit: ")
        if user_input in ['s', 'q']:
            for transition in current_episode:
                self.replay_buffer.add(*transition) # Add demo steps to replay buffer
            print("Added 1 demo to the replay buffer. Current number of steps in the buffer:", len(self.replay_buffer))
            # save GIF of demos
            print('Saving videos (GIFs) of demo...')
            self.record_gif(hand_imgs_for_gif, os.path.join(self.hvideo_dir, f'{episode_num}.gif'))
            self.record_gif(third_person_imgs_for_gif, os.path.join(self.tpvideo_dir, f'{episode_num}.gif'))
        else:
            print("Discarded the demo!")
            print('Saving failed videos...')
            self.record_gif(hand_imgs_for_gif, os.path.join(self.fp_failed_video_dir, f'{episode_num}.gif'))
            self.record_gif(third_person_imgs_for_gif, os.path.join(self.tp_failed_video_dir, f'{episode_num}.gif'))
        return user_input

    def run(self):
        # assumes all episodes are of max length
        episode_num = int(len(self.replay_buffer) / self.max_length) + 1
        assert not self.replay_buffer.full, "The replay buffer is already full!"
        num_demos_remaining =  math.floor((self.replay_buffer.capacity - len(self.replay_buffer)) / self.max_length) # i.e., floor[(capacity_in_steps - num_steps_currently_in_buffer) / (max_num_steps_per_episode)]
        while num_demos_remaining > 0:
            user_input = self.single_demo(episode_num)
            if user_input == 's': # save episode
                episode_num += 1
            if user_input == 'd': # discard episode
                pass
            if user_input == 'q': # save episode and quit
                break

        print('Saving demos and exporting buffer to pickle file...')
        self.replay_buffer.save(self.demo_dir)
        with gzip.open(self.rb_pkl, 'wb') as f:
            pkl.dump(self.replay_buffer, f, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    base_dir = Path('/home/panda5/franka_demos/franka_demos')
    work_dir = base_dir / sys.argv[1]
    workspace = Workspace(work_dir=work_dir)
    workspace.run()