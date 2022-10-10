'''
Basic Robot Environment Wrapper
Robot Specific Functions: self._update_pose(), self.get_ee_pos(), self.get_ee_angle()
Camera Specific Functions: self.render_obs()
Experiment Specific Functions: self.get_info(), self.get_reward(), self.get_observation()
'''
import numpy as np
import time
import gym

from transformations import add_angles, angle_diff
from camera_utils.multi_camera_wrapper import MultiCameraWrapper
from server.robot_interface import RobotInterface
from gym.spaces import Box, Dict

class RobotEnv(gym.Env):
    '''
    Main interface to interact with the robot.
    '''
    def __init__(self,
                 # control frequency
                 hz=10,
                 # randomize arm position on reset  
                 randomize_ee_on_reset=True,
                 # allows user to pause to reset reset of the environment
                 pause_after_reset=False,
                 # observation space configuration
                 hand_centric_view=True, 
                 third_person_view=True,
                 qpos=True,
                 ee_pos=True,
                 # pass IP if not running on NUC
                 ip_address=None,
                 # for state only experiments
                 goal_state=None,
                 # specify path length if resetting after a fixed length
                 max_path_length=None,
                 # use local cameras, else use images from NUC
                 local_cameras=False):

        # Initialize Gym Environment
        super().__init__()

        # Physics
        self.use_desired_pose = True
        self.max_lin_vel = 0.1
        self.max_rot_vel = 0.5
        self.DoF = 3
        self.hz = hz

        self._episode_count = 0
        self._max_path_length = max_path_length

        # reward config, relevant only for state only experiments
        self._goal_state = None
        if goal_state == 'left_open':
            self._goal_state = [1, -1, 1, 1]
        elif goal_state == 'right_closed':
            self._goal_state = [1, 1, 1, -1]

        # resetting configuration
        self._randomize_ee_on_reset = randomize_ee_on_reset
        self._pause_after_reset = pause_after_reset
        self._reset_joint_qpos = np.array([0, 0.115, 0, -2.257, 0.013, 2.257, 1.544])

        # observation space config
        self._first_person = hand_centric_view
        self._third_person = third_person_view
        self._qpos = qpos
        self._ee_pos = ee_pos

        # action space
        self.action_space = Box(
            np.array([-1, -1, -1, -1]), # dx_low, dy_low, dz_low, dgripper_low
            np.array([1, 1, 1, 1]), # dx_high, dy_high, dz_high, dgripper_high
        )
        # EE position (x, y, z) + gripper width
        self.ee_space = Box(
            np.array([0.4, -0.18, 0.16, 0.0045]),
            np.array([0.7, 0.17, 0.3, 0.085]),
        )
        # joint limits + gripper
        self._jointmin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0045], dtype=np.float32)
        self._jointmax = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.085], dtype=np.float32)
        # joint space + gripper
        self.qpos_space = Box(
            self._jointmin,
            self._jointmax
        )

        # final observation space configuration
        env_obs_spaces = {
            'hand_img_obs': Box(0, 255, (3, 100, 100), np.uint8),
            'third_person_img_obs': Box(0, 255, (3, 100, 100), np.uint8),
            'lowdim_ee': self.ee_space,
            'lowdim_qpos': self.qpos_space,
        }
        if not self._first_person:
            env_obs_spaces.pop('hand_img_obs', None)
        if not self._third_person:
            env_obs_spaces.pop('third_person_img_obs', None)
        if not self._qpos:
            env_obs_spaces.pop('lowdim_qpos', None)
        if not self._ee_pos:
            env_obs_spaces.pop('lowdim_ee', None)
        self.observation_space = Dict(env_obs_spaces)
        print(f'configured observation space: {self.observation_space}')

        # robot configuration
        if ip_address is None:
            from franka.robot import FrankaRobot
            self._robot = FrankaRobot(control_hz=self.hz)
        else:
            self._robot = RobotInterface(ip_address=ip_address)

        self._use_local_cameras = local_cameras
        if self._use_local_cameras:
           self._camera_reader = MultiCameraWrapper()

    def step(self, action):
        start_time = time.time()

        assert len(action) == (self.DoF + 1)
        assert (action.max() <= 1) and (action.min() >= -1)

        pos_action, angle_action, gripper = self._format_action(action)
        lin_vel, rot_vel = self._limit_velocity(pos_action, angle_action)
        desired_pos = self._curr_pos + lin_vel
        desired_angle = add_angles(rot_vel, self._curr_angle)
        self._update_robot(desired_pos, desired_angle, gripper)

        comp_time = time.time() - start_time
        sleep_left = max(0, (1 / self.hz) - comp_time)
        time.sleep(sleep_left)
        obs = self.get_observation()

        # reward defaults to 0., unless a goal is specified
        reward = -np.linalg.norm(obs['state'] - self.goal_state) if self._goal_state is not None else 0.
        self._curr_path_length += 1
        done = False
        if self._max_path_length is not None and self._curr_path_length >= self._max_path_length:
            done = True
        return obs, reward, done, {}

    def normalize_ee_obs(self, obs):
        """Normalizes low-dim obs between [-1,1]."""
        # The formula to do this is
        # x_new = 2 * (x - min(x)) / (max(x) - min(x)) - 1
        # x = (x_new + 1) * (max (x) - min(x)) / 2
        # Source: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        normalized_obs = 2 * (obs - self.ee_space.low) / (self.ee_space.high - self.ee_space.low) - 1
        return normalized_obs

    def normalize_qpos(self, qpos):
        """Normalizes qpos between [-1,1]."""
        # The ranges for the joint limits are taken from
        # the franka emika page: https://frankaemika.github.io/docs/control_parameters.html
        norm_qpos = 2 * (qpos - self.qpos_space.low) / (self.qpos_space.high - self.qpos_space.low) - 1
        return norm_qpos

    def reset_gripper(self):
        self._robot.update_gripper(0)
        
    def reset(self):
        self._curr_path_length = 0
        self._robot.update_gripper(0)
        self._robot.update_joints(self._reset_joint_qpos)

        # fix default angle at first joint reset
        if self._episode_count == 0:                              
            self._default_angle = self._robot.get_ee_angle()

        if self._randomize_ee_on_reset:
            self._desired_pose = {'position': self._robot.get_ee_pos(),
                                  'angle': self._robot.get_ee_angle(),
                                  'gripper': 0}
            self._randomize_reset_pos()
            time.sleep(1)

        if self._pause_after_reset:
            user_input = input("Enter (s) to wait 5 seconds & anything else to continue: ")
            if user_input in ['s', 'S']:
                time.sleep(5)

        # initialize desired pose correctly for env.step
        self._desired_pose = {'position': self._robot.get_ee_pos(),
                              'angle': self._robot.get_ee_angle(),
                              'gripper': 0}
        self._episode_count += 1

        return self.get_observation()

    def _format_action(self, action):
        '''Returns [x,y,z], [yaw, pitch, roll], close_gripper'''
        default_delta_angle = angle_diff(self._default_angle, self._curr_angle)
        if self.DoF == 3:
            delta_pos, delta_angle, gripper = action[:-1], default_delta_angle, action[-1]
        elif self.DoF == 6:
            delta_pos, delta_angle, gripper = action[:3], action[3:6], action[-1]
        return np.array(delta_pos), np.array(delta_angle), gripper

    def _limit_velocity(self, lin_vel, rot_vel):
        """Scales down the linear and angular magnitudes of the action"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        if lin_vel_norm > self.max_lin_vel:
            lin_vel = lin_vel * self.max_lin_vel / lin_vel_norm
        if rot_vel_norm > self.max_rot_vel:
            rot_vel = rot_vel * self.max_rot_vel / rot_vel_norm
        lin_vel, rot_vel = lin_vel / self.hz, rot_vel / self.hz
        return lin_vel, rot_vel

    def _update_robot(self, pos, angle, gripper, clip=True):
        """
        input: the commanded position (this will be clipped)
        Feasible position (based on forward kinematics) is tracked and used for updating,
        but the real position is used in observation.
        """
        # clip commanded position to satisfy box constraints
        if clip:
            x_low, y_low, z_low, _ = self.ee_space.low
            x_high, y_high, z_high, _ = self.ee_space.high
            pos[0] = pos[0].clip(x_low, x_high) # new x
            pos[1] = pos[1].clip(y_low, y_high) # new y
            pos[2] = pos[2].clip(z_low, z_high) # new z

        feasible_pos, feasible_angle = self._robot.update_pose(pos, angle)
        self._robot.update_gripper(gripper)
        self._desired_pose = {'position': feasible_pos, 
                              'angle': feasible_angle,
                              'gripper': gripper}

    @property
    def _curr_pos(self):
        if self.use_desired_pose: return self._desired_pose['position'].copy()
        return self._robot.get_ee_pos()

    @property
    def _curr_angle(self):
        if self.use_desired_pose: return self._desired_pose['angle'].copy()
        return self._robot.get_ee_angle()

    def get_images(self):
        camera_feed = []
        if self._use_local_cameras:
            camera_feed.extend(self._camera_reader.read_cameras())
        else:
            camera_feed.extend(self._robot.read_cameras())
        return camera_feed

    def get_state(self):
        state_dict = {}
        gripper_state = self._robot.get_gripper_state()

        state_dict['control_key'] = 'desired_pose' if \
            self.use_desired_pose else 'current_pose'

        state_dict['desired_pose'] = np.concatenate(
            [self._desired_pose['position'],
            self._desired_pose['angle'],
            [self._desired_pose['gripper']]])

        state_dict['current_pose'] = np.concatenate(
            [self._robot.get_ee_pos(),
            self._robot.get_ee_angle(),
            [gripper_state]])

        state_dict['joint_positions'] = self._robot.get_joint_positions()
        state_dict['joint_velocities'] = self._robot.get_joint_velocities()
        # don't track gripper velocity
        state_dict['gripper_velocity'] = 0

        return state_dict

    def _randomize_reset_pos(self):
        '''takes random action along x-y plane, no change to z-axis / gripper'''
        random_vec = np.random.uniform(-0.5, 0.5, (2,))
        act_delta = np.concatenate([random_vec, np.zeros((2,))])
        for _ in range(20):
            self.step(act_delta)

    def get_observation(self):
        # get state and images
        current_state = self.get_state()
        current_images = self.get_images()
        
        # set images
        obs_first = current_images[0]['array']
        obs_third = current_images[1]['array']
        # set gripper width
        gripper_width = current_state['current_pose'][-1:]
        # compute and normalize ee/qpos state
        ee_pos = np.concatenate([current_state['current_pose'][:3], gripper_width])
        qpos = np.concatenate([current_state['joint_positions'],  gripper_width])
        normalized_ee_pos = self.normalize_ee_obs(ee_pos)
        normalized_qpos = self.normalize_qpos(qpos)

        obs_dict = {
            'hand_img_obs': obs_first,
            'third_person_img_obs': obs_third,
            'lowdim_ee': normalized_ee_pos,
            'lowdim_qpos': normalized_qpos,
        }

        if not self._first_person:
            obs_dict.pop('hand_img_obs', None)
        if not self._third_person:
            obs_dict.pop('third_person_img_obs', None)
        if not self._qpos:
            obs_dict.pop('lowdim_qpos', None)
        if not self._ee_pos:
            obs_dict.pop('lowdim_ee', None)

        return obs_dict

    def render(self, mode=None):
        if mode == 'video':
            image_obs = self.get_images()
            obs = np.concatenate([image_obs[0]['array'],
                                  image_obs[0]['array']], axis=0)
            return obs
        else:
            return self.get_observation()

    def is_robot_reset(self, epsilon=0.1):
        curr_joints = self._robot.get_joint_positions()
        joint_dist = np.linalg.norm(curr_joints - self.reset_joints)
        return joint_dist < epsilon

    @property
    def num_cameras(self):
        return len(self.get_images())
