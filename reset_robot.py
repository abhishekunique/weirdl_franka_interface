from robot_env import RobotEnv
ip_add = '172.24.68.68'
env = RobotEnv(ip_address=ip_add)
print(f'reset env')
env.reset()