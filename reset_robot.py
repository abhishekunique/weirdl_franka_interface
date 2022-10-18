from robot_env import RobotEnv
stanford_ip = '10.5.82.75'
local_ip = '172.16.0.10'
env = RobotEnv(ip_address=local_ip)
print(f'reset env')
env.reset()