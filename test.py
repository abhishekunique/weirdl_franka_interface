from robot_env import RobotEnv
#from controllers.xbox_controller import XboxController
import numpy as np
import time

#controller = XboxController(DoF=3)
print(f'init env')
ip_add = None#'172.24.68.68'
env = RobotEnv(ip_address=ip_add)
#env = RobotEnv()

STEP_ENV = True
print(f'reset env')
env.reset()

max_steps = 10000
for i in range(max_steps):
	action = np.random.uniform(0, 1, (4, ))
	obs, reward, done, info = env.step(action)
	if (i % 50) == 0:
		print(f'\n resseting ! \n')
		env.reset()