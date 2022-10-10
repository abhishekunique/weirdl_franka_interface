'''
Xbox Controller Class For Teleoperation
'''
import pygame
import numpy as np

pygame.init()
pygame.joystick.init()

class XboxController(object):
	def __init__(self, DoF=6, pos_gain: float = 0.1, orien_gain: float = 15):
		# Initialize Controller
		self.joystick = pygame.joystick.Joystick(0)
		self.joystick.init()

		# Control Parameters
		self.DoF = DoF
		self.pos_gain = pos_gain
		self.orien_gain = orien_gain
		self.threshold = 0.1
		
		# Save Toggle
		self.button_resetted = True
		self.gripper_state = 0

	def _process_toggle(self, toggle):
		if toggle:
			self.gripper_state = 1
		else:
			self.gripper_state = -1

	def get_info(self):
		pygame.event.get()
		# A is 0, reserved for gripper
		# Y is 3
		# B is 1
		# X is 2
		# reset_episode = self.joystick.get_button(15)

		save_trajectory = self.joystick.get_button(1)
		delete_trajectory = self.joystick.get_button(2)

		if save_trajectory or delete_trajectory:
			self.gripper_state = 0
			self.button_resetted = True
		return {'save_trajectory': save_trajectory, 'delete_trajectory': delete_trajectory}

	def get_action(self):
		pygame.event.get()
		
		# XYZ Dimensions
		x = self.joystick.get_axis(1)
		y = self.joystick.get_axis(0)
		z = (self.joystick.get_axis(5) - self.joystick.get_axis(2)) / 2

		# Orientation Dimensions
		yaw = self.joystick.get_axis(4)
		pitch = self.joystick.get_axis(3)
		roll = self.joystick.get_button(4) - self.joystick.get_button(5)

		# Process Gripper Action (A button)
		# Close/Open depending on whether this is pressed down
		self._process_toggle(self.joystick.get_button(0))

		# Process Pose Action
		pose_action = np.array([x, y, z, yaw, pitch, roll], dtype=np.float32)[:self.DoF]
		pose_action[np.abs(pose_action) < self.threshold] = 0.

		# Scale Accordingly
		# Unsure if this does anything in terms of position gain
		pose_action[:3] * self.pos_gain
		pose_action[3:6] * self.orien_gain
		return np.append(pose_action, [self.gripper_state])


if __name__ == "__main__":
	from robot_env import RobotEnv
	from controllers.xbox_controller import XboxController

	controller = XboxController(DoF=3)
	print(f'init env')
	env = RobotEnv(local_cameras=True)

	print(f'reset env')
	env.reset()

	max_steps = 10000
	for _ in range(max_steps):
		obs = env.get_state()
		action = controller.get_action()
		env.step(action)