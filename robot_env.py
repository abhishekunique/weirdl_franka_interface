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
    
    def __init__(self, use_hand_centric_view=True, use_third_person_view=True, ip_address=None, robot=1, random_reset=True, 
                demo_collection_mode=False, hz=10, pause_resets=False, viz=False, goal_state=None,
                path_length=100, state='ee'):
        # Initialize Gym Environment
        super().__init__()

        # Physics
        self.use_desired_pose = True
        self.max_lin_vel = 0.15
        self.max_rot_vel = 0.5
        self.DoF = 3
        self.hz = hz

        self.epcount = 0
        self.max_path_length = path_length
        self._max_episode_steps = self.max_path_length
        # default reset position
        self.resetpos = np.array([0.5, 0, 0.15]) # x y z (position) gripper status
        self.first_person = use_hand_centric_view
        self.third_person = use_third_person_view
        self.demo_collection_mode = demo_collection_mode
        self.random_reset = random_reset
        self.pause_resets = pause_resets
        self.viz = viz
        self.state = state
        ## Action/Observation Space
        self.action_space = Box(
                np.array([-1, -1, -1, -1]), # dx_low, dy_low, dz_low, dgripper_low
                np.array([1, 1, 1, 1]), # dx_high, dy_high, dz_high, dgripper_high
        )
        # x y z gripper (robotiq)
        self.lowdim_space = Box(
                np.array([0.4, -0.18, 0.05, 0.0045]),
                np.array([0.7, 0.17, 0.3, 0.085]),
        )

        # joint limits
        self.jointmax = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.085], dtype=np.float32)
        self.jointmin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0045], dtype=np.float32)

        # total low dim space with joints and gripper included
        self.qpos_space = Box(
            self.jointmin,
            self.jointmax
        )

        env_obs_spaces = {
            'hand_img_obs': Box(0, 255, (3, 100, 100), np.uint8),
            'third_person_img_obs': Box(0, 255, (3, 100, 100), np.uint8),
            'lowdim_obs': self.lowdim_space,
            'lowdim_qpos': self.qpos_space,
        }
        if not self.first_person:
            env_obs_spaces.pop('hand_img_obs', None)
        if not self.third_person:
            env_obs_spaces.pop('third_person_img_obs', None)
        self.observation_space = Dict(env_obs_spaces)

        x_low, y_low, _, _= self.lowdim_space.low
        x_high, y_high, z_high, _ = self.lowdim_space.high
        x_mid, y_mid = (x_low + x_high) / 2, (y_low + y_high) / 2
        self.resetpos[0] = x_mid
        self.resetpos[1] = y_mid
        self.resetpos[2] = z_high - 0.02
        # track so we can raise EE
        # for joint reset logic
        self.z_high = z_high

        # Robot Configuration
        if ip_address is None:
            from franka.robot import FrankaRobot
            self._robot = FrankaRobot(control_hz=self.hz)
        else:
            self._robot = RobotInterface(ip_address=ip_address)

        self.reset_joints = np.array([0, 0.115, 0, -2.257, 0.013, 2.257, 1.544])
        # Create Cameras
        # use local cameras when running env on NUC
        # use robot cameras when running env on workstation
        self._use_local_cameras = True
        self._use_robot_cameras = False
        
        if self._use_local_cameras:
           self._camera_reader = MultiCameraWrapper()

    def step(self, action):
        start_time = time.time()

        # Process Action
        assert len(action) == (self.DoF + 1)
        assert (action.max() <= 1) and (action.min() >= -1)
        print(f'action! {action}')
        pos_action, angle_action, gripper = self._format_action(action)
        lin_vel, rot_vel = self._limit_velocity(pos_action, angle_action)
        print(f'lin vel {lin_vel}')
        desired_pos = self._curr_pos + lin_vel
        desired_angle = add_angles(rot_vel, self._curr_angle)
        print(f'desired angle {desired_angle}')
        print(f'gripper state {self._robot.gripper_width}')
        # Desired position is current position plus
        # the resulting velocity from current action
        self._update_robot(desired_pos, desired_angle, gripper)

        comp_time = time.time() - start_time
        sleep_left = max(0, (1 / self.hz) - comp_time)
        time.sleep(sleep_left)
        obs = self.get_observation()

        # set reward and done to None for now
        reward = done = None
        return obs, reward, done, {}

    def normalize_lowdim_obs(self, obs):
        """Normalizes low-dim obs between [-1,1]."""
        # The formula to do this is
        # x_new = 2 * (x - min(x)) / (max(x) - min(x)) - 1
        # x = (x_new + 1) * (max (x) - min(x)) / 2
        # Source: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        normalized_obs = np.zeros_like(obs)
        normalized_obs = 2 * (obs - self.lowdim_space.low) / (self.lowdim_space.high - self.lowdim_space.low) - 1
        return normalized_obs

    def normalize_qpos(self, qpos):
        """Normalizes qpos between [-1,1]."""
        # The ranges for the joint limits are taken from
        # the franka emika page: https://frankaemika.github.io/docs/control_parameters.html
        norm_qpos = np.zeros_like(qpos)
        norm_qpos = 2 * (qpos - self.jointmin) / (self.jointmax - self.jointmin) - 1
        return norm_qpos

    def reset(self):
        self._robot.update_gripper(0)

        #if self.epcount % 10 == 0:
        self._robot.update_joints(self.reset_joints)

        if self.pause_resets:
            # sleep for one second so I have time to keyboard interrupt if necessary
            user_input = input("Enter (s) to wait 5 seconds & anything else to continue: ")
            if user_input in ['s', 'S']:
                time.sleep(5)
            else:
                # otherwise just continue
                pass

        self._desired_pose = {'position': self._robot.get_ee_pos(),
                              'angle': self._robot.get_ee_angle(),
                              'gripper': 0}
        # fix default angle at first joint reset
        if self.epcount == 0:                              
            self._default_angle = self._desired_pose['angle']

        self.epcount += 1
        #self.go_to_rest()
        return self.get_observation()

    def _format_action(self, action):
        '''Returns [x,y,z], [yaw, pitch, roll], close_gripper'''
        default_delta_angle = angle_diff(self._default_angle, self._curr_angle)
        if self.DoF == 3:
            delta_pos, delta_angle, gripper = action[:-1], default_delta_angle, action[-1]
        # elif self.DoF == 4:
        #     delta_pos, delta_angle, gripper = action[:3], action[3], action[-1]
        #     delta_angle = delta_angle.extend([0,0])
        # elif self.DoF == 5:
        #     delta_pos, delta_angle, gripper = action[:3], action[3:5], action[-1]
        #     delta_angle = delta_angle.append(0)
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
        Takes in desired position and gripper, calls update_pose which 
        also takes desired_pos as input, and then computes a desired joint
        position (qpos) using the DM IK solver. We then run the forward kinematics
        on this qpos to compute the feasible position (i.e. where the robot can actually
        reach) and we set the joints to the desired qpos and return this realisitc
        position accordingly

        NOTE: the curr_pos function will use this desired_pos, and assumes that the robot
        has gotten close enough. However, if there is error in the forward kinematics
        or the joints aren't quite set right, the real position will be different.
        Thus we should not track the desired pose in get observation
        """
        # clip so position is within Box
        if clip:
            # assumes pos is NP array
            x_low, y_low, z_low, _ = self.lowdim_space.low
            x_high, y_high, z_high, _ = self.lowdim_space.high
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
        if self._use_robot_cameras:
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
            [gripper_state[0]]])

        state_dict['joint_positions'] = self._robot.get_joint_positions()
        state_dict['joint_velocities'] = self._robot.get_joint_velocities()
        state_dict['gripper_velocity'] = gripper_state[1]

        return state_dict

    def go_to_rest(self):
        # reset robot, also update desired pose for controller
        print(f'reseting franka pos!')
        # 1 for an open gripper
        curr_pos = np.concatenate([self._robot.get_ee_pos(), [1]])
        curr_lowdim = self.normalize_lowdim_obs(curr_pos)
        if self.random_reset:
            restp = self.resetpos[:].copy()
            random_vec = np.random.uniform(-0.1, 0.1, (3,))
            restp += random_vec
            rest_pos = np.concatenate([restp, [1]])
            rest_lowdim = self.normalize_lowdim_obs(rest_pos)
            # slowly move to rest from
            # current position with small deltas
            pos_delta = rest_lowdim - curr_lowdim
            act_delta = pos_delta / 20
            for idx in range(20):
                print(f'step {idx}')
                print(f'delta {act_delta}')
                self.step(act_delta)
                curr_pos = np.concatenate([self._robot.get_ee_pos(), [1]])
                curr_lowdim = self.normalize_lowdim_obs(curr_pos)
                pos_delta = rest_lowdim - curr_lowdim
                act_delta = pos_delta / 20
            #t0 = time.time()
            # while time.time() - t0 < 2:
            #     self._update_robot(restp, self._default_angle, 0)
            #     time.sleep(1.0)
        else:    
            t0 = time.time()
            while time.time() - t0 < 2:
                self._update_robot(restp, self._default_angle, 0)
                time.sleep(1.0)
        self._desired_pose = {'position': self._robot.get_ee_pos(),
                              'angle': self._robot.get_ee_angle(),
                              'gripper': 0}
    
    def get_observation(self, include_images=True, include_robot_state=True):
        obs_dict = {}
        if include_images:
            obs_dict['images'] = self.get_images()
        if include_robot_state:
            state_dict = self.get_state()
            obs_dict.update(state_dict)

        # normalize state and qpos
        curr_gripper_width = self._robot.gripper_width
        lowdim_obs = np.concatenate([obs_dict['current_pose'][:3], [curr_gripper_width]])
        qpos = np.concatenate([obs_dict['joint_positions'],  [curr_gripper_width]])
        normalized_lowdim_obs = self.normalize_lowdim_obs(lowdim_obs)
        normalized_qpos = self.normalize_qpos(qpos)
        obs_first = obs_dict['images'][0]['array']
        obs_third = obs_dict['images'][1]['array']

        # format first / third person view
        if self.first_person and not self.third_person:
            img_obs = obs_first
        elif self.third_person and not self.first_person:
            img_obs = obs_third
        elif self.third_person and self.first_person:
            img_obs = np.concatenate([obs_first, obs_third], axis=2)
        
        if self.demo_collection_mode:
            obs_dict = {
                'lowdim_obs': normalized_lowdim_obs,
                'hand_img_obs': obs_first,
                'third_person_img_obs': obs_third,
                'lowdim_qpos': normalized_qpos
            }
        elif self.viz:
            # we only want visual observations
            obs_dict = img_obs
        else:
            state = normalized_lowdim_obs if self.state == 'ee' else normalized_qpos
            obs_dict = {
                'imgs': img_obs,
                'state': state
            }
        return obs_dict

    def render(self, mode=None):
        if mode == 'video':
            # render third person view at full quality
            image_obs = self.get_images()
            obs_first = image_obs['images'][0]['array']
            obs_third = image_obs['images'][1]['array']
            obs = np.concatenate([obs_first, obs_third], axis=0)
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
