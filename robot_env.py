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

class RobotEnv(gym.Env):
    
    def __init__(self, ip_address=None):
        # Initialize Gym Environment
        super().__init__()

        # Physics
        self.use_desired_pose = True
        self.max_lin_vel = 0.1
        self.max_rot_vel = 0.5
        self.DoF = 3
        self.hz = 10

        # Robot Configuration
        if ip_address is None:
            from franka.robot import FrankaRobot
            self._robot = FrankaRobot(control_hz=self.hz)
        else:
            self._robot = RobotInterface(ip_address=ip_address)

        # Drawer task
        self.reset_joints = np.array([0., -np.pi/4,  0, -3/4 * np.pi, 0,  np.pi/2, 0.])

        # Create Cameras
        # use local cameras when running env on NUC
        # use robot cameras when running env on workstation
        self._use_local_cameras = True
        self._use_robot_cameras = False
        
        if self._use_local_cameras:
           self._camera_reader = MultiCameraWrapper()

        self.reset()

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


    def reset(self):
        self._robot.update_gripper(0)
        self._robot.update_joints(self.reset_joints)
        self._desired_pose = {'position': self._robot.get_ee_pos(),
                              'angle': self._robot.get_ee_angle(),
                              'gripper': 0}
        self._default_angle = self._desired_pose['angle']
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

    def _update_robot(self, pos, angle, gripper):
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

    def get_observation(self, include_images=True, include_robot_state=True):
        obs_dict = {}
        if include_images:
            obs_dict['images'] = self.get_images()
        if include_robot_state:
            state_dict = self.get_state()
            obs_dict.update(state_dict)
        return obs_dict

    def is_robot_reset(self, epsilon=0.1):
        curr_joints = self._robot.get_joint_positions()
        joint_dist = np.linalg.norm(curr_joints - self.reset_joints)
        return joint_dist < epsilon

    @property
    def num_cameras(self):
        return len(self.get_images())
