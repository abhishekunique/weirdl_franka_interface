from robot_env import RobotEnv
from controllers.xbox_controller import XboxController
import numpy as np
import time

controller = XboxController(DoF=3)
print(f'init env')
ip_add = '172.16.0.2'
env = RobotEnv()
#env = RobotEnv()

STEP_ENV = True
print(f'reset env')
env.reset()

max_steps = 10000
for i in range(max_steps):
	obs = env.get_state()
	action = controller.get_action()
	env.step(action)