from camera_utils.camera_thread import CameraThread
from camera_utils.cv2_camera import gather_cv2_cameras
import time

class MultiCameraWrapper:

	def __init__(self, specific_cameras=None, use_threads=True):
		self._all_cameras = []

		if specific_cameras is not None:
			self._all_cameras.extend(specific_cameras)
		
		all_cameras = gather_cv2_cameras()
		self._all_cameras.extend(all_cameras)
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
