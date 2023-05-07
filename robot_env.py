'''
Basic Robot Environment Wrapper
Robot Specific Functions: self._update_pose(), self.get_ee_pos(), self.get_ee_angle()
Camera Specific Functions: self.render_obs()
Experiment Specific Functions: self.get_info(), self.get_reward(), self.get_observation()
'''
import numpy as np
import time
import gym
import open3d as o3d
import cv2
from transformations import add_angles, angle_diff
from camera_utils.multi_camera_wrapper import MultiCameraWrapper
from server.robot_interface import RobotInterface
from gym.spaces import Box, Dict

from airobot.sensor.camera.camera import Camera
from airobot.utils.common import to_rot_mat

class RobotEnv(gym.Env):
    '''
    Main interface to interact with the robot.
    '''
    def __init__(self,
                 # control frequency
                 hz=10,
                 DoF=3,
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

        # initialize gym environment
        super().__init__()

        # physics
        self.use_desired_pose = False
        self.max_lin_vel = 0.2 # 0.1
        self.max_rot_vel = 2.0 # 0.5
        self.DoF = DoF
        self.hz = hz

        self._episode_count = 0
        self._max_path_length = max_path_length
        self._curr_path_length = 0

        # reward config, relevant only for state only experiments
        self._goal_state = None
        if goal_state == 'left_open':
            self._goal_state = [1, -1, 1, 1]
        elif goal_state == 'right_closed':
            self._goal_state = [1, 1, 1, -1]
        elif goal_state == 'red_sphere':
            self._goal_state = [0.54667562, -0.21923018, 0.16995791, 0.08017046]

        # resetting configuration
        self._peg_insert = True
        self._randomize_ee_on_reset = randomize_ee_on_reset
        self._pause_after_reset = pause_after_reset
        self._gripper_angle = 0.1 if self._peg_insert else 1.544
        self._reset_joint_qpos = np.array([0, 0.423, 0, -1.944, 0.013, 2.219, self._gripper_angle])

        # observation space config
        self._first_person = hand_centric_view
        self._third_person = third_person_view
        self._qpos = qpos
        self._ee_pos = ee_pos

        # action space
        self.action_space = Box(
            np.array([-1] * (self.DoF + 1)), # dx_low, dy_low, dz_low, dgripper_low
            np.array([ 1] * (self.DoF + 1)), # dx_high, dy_high, dz_high, dgripper_high
        )
        # EE position (x, y, z) + gripper width
        if self.DoF == 3:
            self.ee_space = Box(
                np.array([0.38, -0.25, 0.15, 0.00]),
                np.array([0.70, 0.28, 0.35, 0.085]),
            )
        elif self.DoF == 4:
            # EE position (x, y, z) + gripper width
            self.ee_space = Box(
                np.array([0.55, -0.06, 0.15, -1.57, 0.00]),
                np.array([0.73, 0.28, 0.35, 0.0, 0.085]),
            )

        # joint limits + gripperspecific_cameras
        self._jointmin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0045], dtype=np.float32)
        self._jointmax = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.085], dtype=np.float32)
        # joint space + gripper
        self.qpos_space = Box(
            self._jointmin,
            self._jointmax
        )

        # final observation space configuration
        env_obs_spaces = {
            'hand_img_obs': Box(0, 255, (100, 100, 3), np.uint8),
            'third_person_img_obs': Box(0, 255, (100, 100, 3), np.uint8),
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

        self._hook_safety = False
        self._bowl_safety = False
        self._safety = self._hook_safety or self._bowl_safety

        # Camera parameters for 3D
        self.img_height = 480
        self.img_width = 640
        self.cam_ext_mat = None
        self.cam_int_mat = None
        self.cam_int_mat_inv = None
        self.depth_scale = 0.001
        self.depth_min = 0.2
        self.depth_max = 2

    def step(self, action):
        start_time = time.time()

        assert len(action) == (self.DoF + 1)
        assert (action.max() <= 1) and (action.min() >= -1)

        pos_action, angle_action, gripper = self._format_action(action)
        lin_vel, rot_vel = self._limit_velocity(pos_action, angle_action)
        # clipping + any safety corrections for position
        desired_pos, gripper = self._get_valid_pos_and_gripper(self._curr_pos + lin_vel, gripper)
        desired_angle = add_angles(rot_vel, self._curr_angle)
        if self.DoF == 4:
            desired_angle[2] = desired_angle[2].clip(self.ee_space.low[3], self.ee_space.high[3])
        self._update_robot(desired_pos, desired_angle, gripper)

        comp_time = time.time() - start_time
        sleep_left = max(0, (1 / self.hz) - comp_time)
        time.sleep(sleep_left)
        obs = self.get_observation()

        # reward defaults to 0., unless a goal is specified
        reward = -np.linalg.norm(obs['lowdim_ee'] - self._goal_state) if self._goal_state is not None else 0.
        self._curr_path_length += 1
        done = False
        if self._max_path_length is not None and self._curr_path_length >= self._max_path_length:
            done = True
        return obs, reward, done, {'rgb': obs['hand_img_obs'], 
                                   'rgb_aligned': obs['hand_img_obs_aligned'], 
                                   'depth': obs['third_person_img_obs'], 
                                   'lowdim_ee': self.unnormalize_ee_obs(obs['lowdim_ee'])}

    def normalize_ee_obs(self, obs):
        """Normalizes low-dim obs between [-1,1]."""
        # x_new = 2 * (x - min(x)) / (max(x) - min(x)) - 1
        # x = (x_new + 1) * (max (x) - min(x)) / 2 + min(x)
        # Source: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        normalized_obs = 2 * (obs - self.ee_space.low) / (self.ee_space.high - self.ee_space.low) - 1
        return normalized_obs

    def unnormalize_ee_obs(self, obs):
        return (obs + 1) * (self.ee_space.high - self.ee_space.low) / 2 + self.ee_space.low

    def normalize_qpos(self, qpos):
        """Normalizes qpos between [-1,1]."""
        # The ranges for the joint limits are taken from
        # the franka emika page: https://frankaemika.github.io/docs/control_parameters.html
        norm_qpos = 2 * (qpos - self.qpos_space.low) / (self.qpos_space.high - self.qpos_space.low) - 1
        return norm_qpos

    def reset_gripper(self):
        self._robot.update_gripper(0)

    def reset(self):
        # to move safely, always move up to the highest positions before resetting joints
        if self._safety and self._episode_count > 0:
            cur_pos = self.normalize_ee_obs(np.concatenate([self._curr_pos, [0.]]))[:3]

            if self._hook_safety:
                # if inside the hook, push it back
                while -0.04 <= cur_pos[0] <= 0.3 and 0.66 <= cur_pos[1] <= 0.75 and 0.78 <= cur_pos[2] <= 0.98:
                    self.step(np.array([-0.2, 0., 0., -1.]))
                    cur_pos = self.normalize_ee_obs(np.concatenate([self._curr_pos, [0.]]))[:3]

                # push to the left, till its clear of the hook
                while cur_pos[1] >= 0.10:
                    self.step(np.array([0.0, -0.2, 0., 1.]))
                    cur_pos = self.normalize_ee_obs(np.concatenate([self._curr_pos, [0.]]))[:3]

            if self._bowl_safety:
                # raise the gripper and then reset it
                while cur_pos[2] <= 0.5:
                    self.step(np.array([0.0, 0.0, 1.0, 1.]))
                    cur_pos = self.normalize_ee_obs(np.concatenate([self._curr_pos, [0.]]))[:3]

        self.reset_gripper()
        for _ in range(5):
            self._robot.update_joints(self._reset_joint_qpos)
            if self.is_robot_reset():
                break
            else:
                print('reset failed, trying again')

        # fix default angle at first joint reset
        if self._episode_count == 0:                              
            self._default_angle = self._robot.get_ee_angle()

        if self._randomize_ee_on_reset:
            self._desired_pose = {'position': self._robot.get_ee_pos(),
                                  'angle': self._robot.get_ee_angle(),
                                  'gripper': 1}
            self._randomize_reset_pos()
            time.sleep(1)

        if self._pause_after_reset:
            user_input = input("Enter (s) to wait 5 seconds & anything else to continue: ")
            if user_input in ['s', 'S']:
                time.sleep(5)

        # initialize desired pose correctly for env.step
        self._desired_pose = {'position': self._robot.get_ee_pos(),
                              'angle': self._robot.get_ee_angle(),
                              'gripper': 1}

        self._curr_path_length = 0
        self._episode_count += 1

        return self.get_observation()

    def _format_action(self, action):
        '''Returns [x,y,z], [yaw, pitch, roll], close_gripper'''
        default_delta_angle = angle_diff(self._default_angle, self._curr_angle)
        if self.DoF == 3:
            delta_pos, delta_angle, gripper = action[:-1], default_delta_angle, action[-1]
        elif self.DoF == 4:
            delta_pos, delta_angle, gripper = action[:3], [default_delta_angle[0], default_delta_angle[1], action[3]], action[-1]
        elif self.DoF == 6:
            delta_pos, delta_angle, gripper = action[:3], action[3:6], action[-1]
        return np.array(delta_pos), np.array(delta_angle), gripper

    def _limit_velocity(self, lin_vel, rot_vel):
        """Scales down the linear and angular magnitudes of the action"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        if lin_vel_norm > 1:
            lin_vel = lin_vel / lin_vel_norm
        if rot_vel_norm > 1:
            rot_vel = rot_vel / rot_vel_norm
        lin_vel, rot_vel = lin_vel * self.max_lin_vel / self.hz, rot_vel * self.max_rot_vel / self.hz
        return lin_vel, rot_vel

    def _get_valid_pos_and_gripper(self, pos, gripper):
        '''To avoid situations where robot can break the object / burn out joints,
        allowing us to specify (x, y, z, gripper) where the robot cannot enter. Gripper is included
        because (x, y, z) is different when gripper is open/closed.

        There are two ways to do this: (a) reject action and maintain current pose or (b) project back
        to valid space. Rejecting action works, but it might get stuck inside the box if no action can
        take it outside. Projection is a hard problem, as it is a non-convex set :(, but we can follow
        some rough heuristics.'''

        # clip commanded position to satisfy box constraints
        x_low, y_low, z_low = self.ee_space.low[:3]
        x_high, y_high, z_high = self.ee_space.high[:3]
        pos[0] = pos[0].clip(x_low, x_high) # new x
        pos[1] = pos[1].clip(y_low, y_high) # new y
        pos[2] = pos[2].clip(z_low, z_high) # new z

        '''Prevents robot from entering unsafe territory.

        Unsafe cube is specified as a set of constraints:
        [(x_low, y_low, z_low), (x_high, y_high, z_high)]. Whenever, (x, y, z) falls into 
        any of the boxes, constrained is violated. When specifying one-sided constraint,
        use 2.5/-2.5 as the other limit.

        Assumption: you can only violate _exactly_ one constraint at a time.'''
        if self._safety:
            MAX_LIM = 2.5
            # constraints are computed for normalized observation
            cur_pos = self.normalize_ee_obs(np.concatenate([pos, [0.]]))[:3]
            assert np.linalg.norm(pos - self.unnormalize_ee_obs(np.concatenate([cur_pos, [0.]]))[:3] <= 1e-6) 

            unsafe_box = []

            if self._hook_safety:
                # DO NOT open gripper if passing through the hook
                if 0.08 <= cur_pos[0] <= 0.3 and 0.66 <= cur_pos[1] <= 0.75 and 0.78 <= cur_pos[2] <= 0.98:
                    gripper = -1

                # open gripper
                if gripper >= -0.8:
                    # when height is high enough
                    unsafe_box.append(np.array([[-0.22, 0.14, 0.02], [0.36, MAX_LIM, 0.96]]))
                    # wrist camera hits the hook
                    unsafe_box.append(np.array([[-0.66, 0.14, -MAX_LIM], [0.58, MAX_LIM, 0.02]]))
                # gripper closed
                else:
                    # tunnel for cloth draping
                    unsafe_box.append(np.array([[-0.22, 0.14, 0.78], [0.36, 0.66, MAX_LIM]]))
                    unsafe_box.append(np.array([[-0.22, 0.75, 0.78], [0.36, MAX_LIM, MAX_LIM]]))
                    # height is high enough not to risk wrist camera hitting the hook, so can come closer
                    unsafe_box.append(np.array([[-0.22, 0.14, 0.05], [0.36, MAX_LIM, 0.78]]))
                    # wrist camera hits the hook
                    unsafe_box.append(np.array([[-0.66, 0.14, -MAX_LIM], [0.58, MAX_LIM, 0.05]]))

            elif self._bowl_safety and False:
                width_out = 0.04
                width = 0.04
                # approximate outside dimensions
                y1_out, y2_out = -0.64, 0.69
                x1_out, x2_out = -0.45, 0.66
                z_out = 0.21
                z_out_g = 0.45
                # approximate inside dimensions, y varies based on whether gripper is closed or open
                y1_in_g, y2_in_g = -0.20, 0.25
                y1_in, y2_in = -0.06, 0.12
                x1_in, x2_in = -0.06, 0.42
                z_in = -0.35
                z_in_g = -0.4

                if gripper >= 0.8:
                    # outside, y-axis boxes
                    unsafe_box.append(np.array([[x1_out, y1_out - width_out, -MAX_LIM], [x2_out, y1_out, z_out_g]]))
                    unsafe_box.append(np.array([[x1_out, y2_out, -MAX_LIM], [x2_out, y2_out + width_out, z_out_g]]))
                    # outside, x-axis boxes
                    unsafe_box.append(np.array([[x1_out - width_out, y1_out, -MAX_LIM], [x1_out, y2_out, z_out_g]]))
                    unsafe_box.append(np.array([[x2_out, y1_out, -MAX_LIM], [x2_out + width_out, y2_out, z_out_g]]))
                    # inside, y-axis boxes
                    unsafe_box.append(np.array([[x1_in, y1_in_g, -MAX_LIM], [x2_in, y1_in_g + width, z_in_g]]))
                    unsafe_box.append(np.array([[x1_in, y2_in_g - width, -MAX_LIM], [x2_in, y2_in_g, z_in_g]]))
                    # inside, x-axis boxes
                    unsafe_box.append(np.array([[x1_in, y1_in_g, -MAX_LIM], [x1_in + width, y2_in_g, z_in_g]]))
                    unsafe_box.append(np.array([[x2_in - width, y1_in_g, -MAX_LIM], [x2_in, y2_in_g, z_in_g]]))
                else:
                    # outside, y-axis boxes
                    unsafe_box.append(np.array([[x1_out, y1_out - width_out, -MAX_LIM], [x2_out, y1_out, z_out]]))
                    unsafe_box.append(np.array([[x1_out, y2_out, -MAX_LIM], [x2_out, y2_out + width_out, z_out]]))
                    # outside, x-axis boxes
                    unsafe_box.append(np.array([[x1_out - width_out, y1_out, -MAX_LIM], [x1_out, y2_out, z_out]]))
                    unsafe_box.append(np.array([[x2_out, y1_out, -MAX_LIM], [x2_out + width_out, y2_out, z_out]]))
                    # inside, y-axis boxes
                    unsafe_box.append(np.array([[x1_in, y1_in, -MAX_LIM], [x2_in, y1_in + width, z_in]]))
                    unsafe_box.append(np.array([[x1_in, y2_in - width, -MAX_LIM], [x2_in, y2_in, z_in]]))
                    # inside, x-axis boxes
                    unsafe_box.append(np.array([[x1_in, y1_in, -MAX_LIM], [x1_in + width, y2_in, z_in]]))
                    unsafe_box.append(np.array([[x2_in - width, y1_in, -MAX_LIM], [x2_in, y2_in, z_in]]))

            def _violate(pos):
                for _, constraint in enumerate(unsafe_box):
                    if np.min(np.concatenate([pos - constraint[0], constraint[1] - pos])) >= 0.:
                        return True
                return False

            '''It is possible that it is still violating the constraint after the first projection.
            Keep trying different axes till it is not violating ANY constraint.'''
            for constraint in unsafe_box:
                slacks = np.concatenate([cur_pos - constraint[0], constraint[1] - cur_pos])
                if np.min(slacks) >= 0.:
                    print('too close to the hook!')
                    num_tries = 0
                    while _violate(cur_pos) and num_tries < 3:
                        # reset pos before trying another dimension
                        cur_pos = self.normalize_ee_obs(np.concatenate([pos, [0.]]))[:3]
                        min_idx = np.argmin(slacks)
                        if min_idx // 3:
                            cur_pos[min_idx % 3] +=  (slacks[min_idx] + 1e-2)
                        else:
                            cur_pos[min_idx % 3] -= (slacks[min_idx] + 1e-2)
                        slacks[min_idx % 3] = slacks[min_idx % 3 + 3] = np.inf
                        num_tries += 1
                    return self.unnormalize_ee_obs(np.concatenate([cur_pos, [0.]]))[:3], gripper

        return pos, gripper

    def _update_robot(self, pos, angle, gripper):
        """input: the commanded position (clipped before).
        feasible position (based on forward kinematics) is tracked and used for updating,
        but the real position is used in observation."""
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
        random_xy = np.random.uniform(-0.5, 0.5, (2,))
        random_z = np.random.uniform(-0.2, 0.2, (1,))
        if self.DoF == 4:
            random_rot = np.random.uniform(-0.5, 0., (1,))
            act_delta = np.concatenate([random_xy, random_z, random_rot, np.zeros((1,))])
        else:
            act_delta = np.concatenate([random_xy, random_z, np.zeros((1,))])
        for _ in range(10):
            self.step(act_delta)

    def get_observation(self):
        # get state and images
        current_state = self.get_state()
        current_images = self.get_images()

        # set images
        obs_first = current_images[0]['color_image']
        obs_third = current_images[1]['depth_image']

        # set images
        aligned_obs_first = current_images[0]['array']
        
        # set gripper width
        gripper_width = current_state['current_pose'][-1:]
        # compute and normalize ee/qpos state
        if self.DoF == 3:
            ee_pos = np.concatenate([current_state['current_pose'][:3], gripper_width])
        elif self.DoF == 4:
            ee_pos = np.concatenate([current_state['current_pose'][:3], current_state['current_pose'][5:6], gripper_width])
        qpos = np.concatenate([current_state['joint_positions'],  gripper_width])
        normalized_ee_pos = self.normalize_ee_obs(ee_pos)
        normalized_qpos = self.normalize_qpos(qpos)

        color_image_small = cv2.resize(obs_first, dsize=(128, 96), interpolation=cv2.INTER_AREA)
        depth_colormap_small = cv2.resize(obs_third, dsize=(128, 96), interpolation=cv2.INTER_AREA)

        obs_dict = {
            'hand_img_obs': obs_first,
            'hand_img_obs_aligned': aligned_obs_first,
            'third_person_img_obs': obs_third,
            'hand_img_obs_small': color_image_small,
            'third_person_img_obs_small': depth_colormap_small,
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
        # TODO: update rendering to use height, width (for high quality evaluation rendering)
        if mode == 'video':
            image_obs = self.get_images()
            obs = np.concatenate([image_obs[0]['array'],
                                  image_obs[1]['array']], axis=0)
            return obs
        else:
            return self.get_observation()

    def is_robot_reset(self, epsilon=0.1):
        curr_joints = self._robot.get_joint_positions()
        joint_dist = np.linalg.norm(curr_joints - self._reset_joint_qpos)
        return joint_dist < epsilon

    @property
    def num_cameras(self):
        return len(self.get_images())

    def _init_pers_mat(self):
        """
        Initialize related matrices for projecting
        pixels to points in camera frame.
        """
        self.cam_int_mat_inv = np.linalg.inv(self.cam_int_mat)

        img_pixs = np.mgrid[0: self.img_height,
                            0: self.img_width].reshape(2, -1)
        img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
        self._uv_one = np.concatenate((img_pixs,
                                       np.ones((1, img_pixs.shape[1]))))
        self._uv_one_in_cam = np.dot(self.cam_int_mat_inv, self._uv_one)

    def get_cam_ext(self):
        """
        Return the camera's extrinsic matrix.

        Returns:
            np.ndarray: extrinsic matrix (shape: :math:`[4, 4]`)
            for the camera (source frame: base frame.
            target frame: camera frame).
        """
        return self.cam_ext_mat

    def get_cam_int(self, camera_id):
        """
        Return the camera's intrinsic matrix.

        Returns:
            np.ndarray: intrinsic matrix (shape: :math:`[3, 3]`)
            for the camera.
        """
        # TODO: Make this general
        # intr = self._camera_reader._all_cameras[camera_id]._camera.intr
        # cam_int_mat =  np.array([[intr.fx, 0., intr.ppx],
        #                                 [0., intr.fy, intr.ppy],
        #                                 [0., 0.,  1.]])
        cam_int_mat = np.array([[391.59536743,   0.        , 321.44470215],
                            [  0.        , 391.59536743, 239.72434998],
                            [  0.        ,   0.        ,   1.        ]])

        return cam_int_mat
    
    def set_int_matrix(self, fx, fy, ppx, ppy, coeff=None):
        # consider handle distortion here
        self.cam_int_mat =  np.array([[fx, 0., ppx],
                                        [0., fy, ppy],
                                        [0., 0.,  1.]])

    def set_ext_matrix(self, ori_mat, pos):
        cam_mat = np.eye(4)
        cam_mat[:3, :3] = ori_mat
        cam_mat[:3, 3] = pos.flatten()        
        self.cam_ext_mat = cam_mat

    def get_pix_3dpt(self, rs, cs, in_world=True, filter_depth=False,
                     k=1, ktype='median', depth_min=None, depth_max=None,
                     cam_ext_mat=None):
        """
        Calculate the 3D position of pixels in the RGB image.

        Args:
            rs (int or list or np.ndarray): rows of interest.
                It can be a list or 1D numpy array
                which contains the row indices. The default value is None,
                which means all rows.
            cs (int or list or np.ndarray): columns of interest.
                It can be a list or 1D numpy array
                which contains the column indices. The default value is None,
                which means all columns.
            in_world (bool): if True, return the 3D position in
                the world frame,
                Otherwise, return the 3D position in the camera frame.
            filter_depth (bool): if True, only pixels with depth values
                between [depth_min, depth_max]
                will remain.
            k (int): kernel size. A kernel (slicing window) will be used
               to get the neighboring depth values of the pixels specified
               by rs and cs. And depending on the ktype, a corresponding
               method will be applied to use some statistical value
               (such as minimum, maximum, median, mean) of all the depth
               values in the slicing window as a more robust estimate of
               the depth value of the specified pixels.
            ktype (str): what kind of statistical value of all the depth
               values in the sliced kernel
               to use as a proxy of the depth value at specified pixels.
               It can be `median`, `min`, `max`, `mean`.
            depth_min (float): minimum depth value. If None, it will use the
                default minimum depth value defined in the config file.
            depth_max (float): maximum depth value. If None, it will use the
                default maximum depth value defined in the config file.
            cam_ext_mat (np.ndarray): camera extrinsic matrix (shape: :math:`[4,4]`).
                If provided, it will be used to compute the points in the world frame.

        Returns:
            np.ndarray: 3D point coordinates of the pixels in
            camera frame (shape: :math:`[N, 3]`).
        """
        if not isinstance(rs, int) and not isinstance(rs, list) and \
                not isinstance(rs, np.ndarray):
            raise TypeError('rs should be an int, a list or a numpy array')
        if not isinstance(cs, int) and not isinstance(cs, list) and \
                not isinstance(cs, np.ndarray):
            raise TypeError('cs should be an int, a list or a numpy array')
        if isinstance(rs, int):
            rs = [rs]
        if isinstance(cs, int):
            cs = [cs]
        if isinstance(rs, np.ndarray):
            rs = rs.flatten()
        if isinstance(cs, np.ndarray):
            cs = cs.flatten()
        if not (isinstance(k, int) and (k % 2) == 1):
            raise TypeError('k should be a positive odd integer.')
        _, depth_im = self.get_images(get_rgb=False, get_depth=True)
        if k == 1:
            depth_im = depth_im[rs, cs]
        else:
            depth_im_list = []
            if ktype == 'min':
                ktype_func = np.min
            elif ktype == 'max':
                ktype_func = np.max
            elif ktype == 'median':
                ktype_func = np.median
            elif ktype == 'mean':
                ktype_func = np.mean
            else:
                raise TypeError('Unsupported ktype:[%s]' % ktype)
            for r, c in zip(rs, cs):
                s = k // 2
                rmin = max(0, r - s)
                rmax = min(self.img_height, r + s + 1)
                cmin = max(0, c - s)
                cmax = min(self.img_width, c + s + 1)
                depth_im_list.append(ktype_func(depth_im[rmin:rmax,
                                                cmin:cmax]))
            depth_im = np.array(depth_im_list)

        depth = depth_im.reshape(-1) * self.depth_scale
        img_pixs = np.stack((rs, cs)).reshape(2, -1)
        img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
        depth_min = depth_min if depth_min else self.depth_min
        depth_max = depth_max if depth_max else self.depth_max
        if filter_depth:
            valid = depth > depth_min
            valid = np.logical_and(valid,
                                   depth < depth_max)
            depth = depth[:, valid]
            img_pixs = img_pixs[:, valid]
        uv_one = np.concatenate((img_pixs,
                                 np.ones((1, img_pixs.shape[1]))))
        uv_one_in_cam = np.dot(self.cam_int_mat_inv, uv_one)
        pts_in_cam = np.multiply(uv_one_in_cam, depth)
        if in_world:
            if self.cam_ext_mat is None and cam_ext_mat is None:
                raise ValueError('Please call set_cam_ext() first to set up'
                                 ' the camera extrinsic matrix')
            cam_ext_mat = self.cam_ext_mat if cam_ext_mat is None else cam_ext_mat
            pts_in_cam = np.concatenate((pts_in_cam,
                                         np.ones((1, pts_in_cam.shape[1]))),
                                        axis=0)
            pts_in_world = np.dot(cam_ext_mat, pts_in_cam)
            pts_in_world = pts_in_world[:3, :].T
            return pts_in_world
        else:
            return pts_in_cam.T

    # def get_pcd(self, in_world=True, filter_depth=True,
    #             depth_min=None, depth_max=None, cam_ext_mat=None,
    #             rgb_image=None, depth_image=None):
    #     """
    #     Get the point cloud from the entire depth image
    #     in the camera frame or in the world frame.

    #     Args:
    #         in_world (bool): return point cloud in the world frame, otherwise,
    #             return point cloud in the camera frame.
    #         filter_depth (bool): only return the point cloud with depth values
    #             lying in [depth_min, depth_max].
    #         depth_min (float): minimum depth value. If None, it will use the
    #             default minimum depth value defined in the config file.
    #         depth_max (float): maximum depth value. If None, it will use the
    #             default maximum depth value defined in the config file.
    #         cam_ext_mat (np.ndarray): camera extrinsic matrix (shape: :math:`[4,4]`).
    #             If provided, it will be used to compute the points in the world frame.
    #         rgb_image (np.ndarray): externally captured RGB image, if we want to
    #             convert a depth image captured outside this function to a point cloud.
    #             (shape :math:`[H, W, 3]`)
    #         depth_image (np.ndarray): externally captured depth image, if we want to
    #             convert a depth image captured outside this function to a point cloud.
    #             (shape :math:`[H, W]`)

    #     Returns:
    #         2-element tuple containing

    #         - np.ndarray: point coordinates (shape: :math:`[N, 3]`).
    #         - np.ndarray: rgb values (shape: :math:`[N, 3]`).
    #     """
    #     if depth_image iget_pcds None or rgb_image is None:
    #         # rgb_im, depth_im = self.get_images()
    #         images = self.get_images()
    #         rgb_im, depth_im = images[0]["array"], images[1]["array"]
    #     else:
    #         rgb_im = rgb_image
    #         depth_im = depth_image
    #     # pcd in camera from depth
    #     depth = depth_im.reshape(-1) * self.depth_scale
    #     rgb = None
    #     if rgb_im is not None:
    #         rgb = rgb_im.reshape(-1, 3)
    #     depth_min = depth_min if depth_min else self.depth_min
    #     depth_max = depth_max if depth_max else self.depth_max
    #     if filter_depth:
    #         valid = depth > depth_min
    #         valid = np.logical_and(valid,
    #                                depth < depth_max)
    #         depth = depth[valid]
    #         if rgb is not None:
    #             rgb = rgb[valid]
    #         uv_one_in_cam = self._uv_one_in_cam[:, valid]
    #     else:
    #         uv_one_in_cam = self._uv_one_in_cam
    #     pts_in_cam = np.multiply(uv_one_in_cam, depth)
    #     if not in_world:
    #         pcd_pts = pts_in_cam.T
    #         pcd_rgb = rgb
    #         return pcd_pts, pcd_rgb
    #     else:
    #         if self.cam_ext_mat is None and cam_ext_mat is None:
    #             raise ValueError('Please call set_cam_ext() first to set up'
    #                              ' the camera extrinsic matrix')
    #         cam_ext_mat = self.cam_ext_mat if cam_ext_mat is None else cam_ext_mat
    #         pts_in_cam = np.concatenate((pts_in_cam,
    #                                      np.ones((1, pts_in_cam.shape[1]))),
    #                                     axis=0)
    #         pts_in_world = np.dot(cam_ext_mat, pts_in_cam)
    #         pcd_pts = pts_in_world[:3, :].T
    #         pcd_rgb = rgb
    #         return pcd_pts, pcd_rgb
    
    def get_pcd(self, camera_id, depth_scale=1., depth_max=np.inf, return_numpy=True):
        """
        Read images from camera and convert to open3d / numpy pointcloud

        Args
            int: depth_max: maximum depth value to threshold pointcloud
            bool: return_numpy: whether to return pointcloud as np.ndarray or open3D pointcloud
        Returns
            np.ndarray: points
            OR
            o3d.geometry.PointCloud
        """
        imgs = self.get_images()

        rgb_raw = imgs[0]["array"]
        # # use unscaled depth
        d_raw = imgs[1]["depth_image"]
        # import pickle
        # rgb_raw = pickle.load(open('/home/siri/Projects/minimal-rl/test_img_rgb_calib.pkl', 'rb'))
        # d_raw = pickle.load(open('/home/siri/Projects/minimal-rl/test_img_depth_calib.pkl', 'rb'))
        rgb = o3d.geometry.Image(rgb_raw)
        d = o3d.geometry.Image(d_raw)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, d, depth_scale=depth_scale, depth_trunc=depth_max, convert_rgb_to_intensity=False)

        height = np.asarray(rgb).shape[0]
        width = np.asarray(rgb).shape[1]

        cam_int = self.get_cam_int(camera_id)
        cam_ext = self.get_cam_ext()

        import IPython
        IPython.embed()
        
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, cam_int[0,0], cam_int[1,1], cam_int[0,2], cam_int[1,2])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic=intrinsics, extrinsic=np.linalg.inv(cam_ext))

        if return_numpy:
            return np.asarray(pcd.points)
        else:
            return pcd
        
    # env.visualize_rgb_pc(rgb=obs['hand_img_obs'], depth=obs['third_person_img_obs'],
    #                      intrinsics=env.cam_int_mat, extrinsics=env.cam_ext_mat, animation=True)
    def visualize_rgb_pc(self, rgb, depth, fx, fy, ppx, ppy, intrinsics, extrinsics, animation=True):
        height, width = depth.shape
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
        py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
        points = np.float32([px, py, depth]).transpose(1, 2, 0)
        padding = ((0, 0), (0, 0), (0, 1))
        homogen_points = np.pad(points.copy(), padding,
                                'constant', constant_values=1)
        for i in range(3):
            points[Ellipsis, i] = np.sum(extrinsics[i, :] * homogen_points, axis=-1)

        pcd = o3d.geometry.PointCloud()
        pts = np.array(points).reshape(-1, 3)
        clr = np.array(rgb).reshape(-1, 3) / 255
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(clr)
        o3d.visualization.draw_geometries([pcd],
                                            # zoom=0.3412,
                                            # front=[0.4257, -0.2125, -0.8795],
                                            # lookat=[2.6172, 2.0475, 1.532],
                                            # up=[-0.0694, -0.9768, 0.2024])
        )
        return (points, rgb/255.)
    
    def visualize_pcd(self, pcd):
        """
        Visualizes an open3D pointcloud in an interative window.
        """
        o3d.open3d.visualization.draw_geometries([pcd])

    def moveToJointPosition(self, joint_config):
        self._robot.update_joints(np.asarray(joint_config))
        
    def goHome(self):
        home_config = [0.0, -np.pi/4, 0.0, -2*np.pi/3, 0.0, np.pi/3, np.pi/4]
        self.moveToJointPosition(home_config)