from robot_env import RobotEnv
#from controllers.xbox_controller import XboxController
import numpy as np
import time

#controller = XboxController(DoF=3)
print(f'init env')
ip_add = '172.24.68.68'
env = RobotEnv(ip_address=ip_add)
#env = RobotEnv()

STEP_ENV = True
print(f'reset env')
env.reset()

max_steps = 10000
for i in range(max_steps):
	obs = env.get_state()
	action = np.random.uniform(0, 0.5, (4, ))
	env.step(action)