from robot_env import RobotEnv
local_ip = '172.16.0.10'
env = RobotEnv(ip_address=local_ip, randomize_ee_on_reset=False)
print(f'reset env')
env.reset()