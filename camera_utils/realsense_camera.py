import pyrealsense2 as rs
import numpy as np
import time
import cv2

def gather_realsense_cameras():
	context = rs.context()
	
	# # https://github.com/IntelRealSense/librealsense/issues/6628#issuecomment-646558144
	# devices = context.query_devices()
	# for dev in devices:
	# 	dev.hardware_reset()

	all_devices = list(context.devices)
	all_rs_cameras = []

	for device in all_devices:
		rs_camera = RealSenseCamera(device)
		all_rs_cameras.append(rs_camera)

	return all_rs_cameras

class RealSenseCamera:
	def __init__(self, device):
		self._pipeline = rs.pipeline()
		self._serial_number = str(device.get_info(rs.camera_info.serial_number))
		config = rs.config()

		config.enable_device(self._serial_number)

		config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
		device_product_line = str(device.get_info(rs.camera_info.product_line))
		
		if device_product_line == 'L500': config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
		else: config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

		self.align = rs.align(rs.stream.depth)

		cfg = self._pipeline.start(config)
		self.intr = cfg.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

	def read_camera(self, enforce_same_dim=False):
		# Wait for a coherent pair of frames: depth and color
		frames = self._pipeline.wait_for_frames()

		color_frame_unaligned = frames.get_color_frame()
		color_image_unaligned = np.asanyarray(color_frame_unaligned.get_data())
		color_image_unaligned = cv2.cvtColor(color_image_unaligned, cv2.COLOR_BGR2RGB)

		frames = self.align.process(frames)

		depth_frame = frames.get_depth_frame()
		color_frame = frames.get_color_frame()
		if not depth_frame or not color_frame: return None
		read_time = time.time()

		# Convert images to numpy arrays
		depth_image = np.asanyarray(depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())

		# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

		# depth_colormap_dim = depth_colormap.shape
		# color_colormap_dim = color_image.shape
		# If depth and color resolutions are different, resize color image to match depth image for display
		# if depth_colormap_dim != color_colormap_dim and enforce_same_dim:
		# 	color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)

		# color_image = cv2.resize(color_image, dsize=(128, 96), interpolation=cv2.INTER_AREA)
		# depth_colormap = cv2.resize(depth_colormap, dsize=(128, 96), interpolation=cv2.INTER_AREA)

		color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
		depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

		dict_1 = {'array': color_image, 'color_image': color_image_unaligned, 'shape': color_image.shape, 'type': 'rgb',
			'read_time': read_time, 'serial_number': self._serial_number + '/rgb'}
		dict_2 = {'array': depth_colormap, 'depth_image': depth_image, 'shape': color_image.shape, 'type': 'depth',
			'read_time': read_time, 'serial_number': self._serial_number + '/depth'}
		
		return [dict_1, dict_2]
		
	def disable_camera(self):
		self._pipeline.stop()
