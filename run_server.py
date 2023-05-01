from franka.robot import FrankaRobot
from camera_utils.multi_camera_wrapper import MultiCameraWrapper
from server.robot_server import start_server
from camera_utils.realsense_camera import gather_realsense_cameras

if __name__ == '__main__':
    robot = FrankaRobot()
    rs_cameras = gather_realsense_cameras()
    cameras = MultiCameraWrapper(specific_cameras=rs_cameras)
    start_server(robot, cameras)