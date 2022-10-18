from robot_env import RobotEnv
ip_add = '10.5.82.75'
env = RobotEnv(ip_address=ip_add)
print(f'reset env')
env.reset()