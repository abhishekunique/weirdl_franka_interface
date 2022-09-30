from camera_utils.camera_thread import CameraThread
#from camera_utils.realsense_camera import gather_realsense_cameras
#from camera_utils.zed_camera import gather_zed_cameras
import time

class MultiCameraWrapper:

	def __init__(self, camera_types=['realsense', 'zed'], specific_cameras=None, use_threads=True):
		self._all_cameras = []

		if specific_cameras is not None:
			self._all_cameras.extend(specific_cameras)

		if 'realsense' in camera_types:
			realsense_cameras = gather_realsense_cameras()
			self._all_cameras.extend(realsense_cameras)

		if 'zed' in camera_types:
			zed_cameras = gather_zed_cameras()
			self._all_cameras.extend(zed_cameras)

		if use_threads:
			for i in range(len(self._all_cameras)):
				self._all_cameras[i] = CameraThread(self._all_cameras[i])
			time.sleep(1)
	
	def read_cameras(self):
		all_frames = []
		for camera in self._all_cameras:
			curr_feed = camera.read_camera()
			# while curr_feed is None:
			# 	curr_feed = camera.read_camera()
			if curr_feed is not None:
				all_frames.extend(curr_feed)
		return all_frames

	def disable_cameras(self):
		for camera in self._all_cameras:
			camera.disable_camera()
