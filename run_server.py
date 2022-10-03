from franka.robot import FrankaRobot
from camera_utils.multi_camera_wrapper import MultiCameraWrapper
from server.robot_server import start_server

if __name__ == '__main__':
    robot = FrankaRobot()
    cameras = MultiCameraWrapper()
    start_server(robot, cameras)