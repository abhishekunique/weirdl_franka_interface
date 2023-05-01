import cv2
import numpy as np
import PIL
import pixellib
from pixellib.semantic import semantic_segmentation
from fairo.perception.sandbox.eyehandcal.src.eyehandcal.utils import build_proj_matrix
from tqdm import tqdm

from tools.interact_tools import SamControler
from tracker.base_tracker import BaseTracker
from inpainter.base_inpainter import BaseInpainter
import numpy as np
import argparse

# Input: a batch of rgbd videos, same resolution and length. Could also take in the resolution/horizon as argument.
# Output: object position in robot frame for each frame in each video.
# What it does:
# 1. For first frame, either use opencv to segment color red, or run semantic segmentation to detect the ball.
# 2. Optionally, compute the center of mass using cv2 moments() to prompt track anything.
# 3. Run track anything (take in a point or a mask) to get segmentation mask for object in each frame.
# 4. Compute the center of mass of object mask in each frame.
# 5. Combine with depth information and map object position to robot frame using the calibration matrix.
# 6. For first frame, either use opencv to segment color red, or run semantic segmentation to detect the ball.
# 7. Optionally, compute the center of mass using cv2 moments() to prompt track anything.
# 8. Run track anything (take in a point or a mask) to get segmentation mask for object in each frame.
# 9. Computer the center of mass of object mask in each frame.
# 10. Combine with depth information and map object position to robot frame using the calibration matrix.

class CamObjWrapper(Wrapper):
    
    def __init__(self, env):
        #self.rgb_videos = 
        

    def get_object_position(self, rgb_videos, resolution, horizon):
        # Here we get it from calibration.json from the fairo perception
        #K = np.array([[385.8476257324219, 0, cx], [0, 385.30584716796875, cy], [0, 0, 1]])
        K = build_proj_matrix(fx=385.8476257324219,  fy=385.30584716796875, ppx=321.3341979980469, ppy=240.64756774902344)
        object_positions = []
        for video in rgb_videos:
            object_positions_video = []
            frame = video[0]
            mask = self.segment_object(frame)
            com = self.get_center_of_mass(mask) # use cv2 moments()

            for i in range(1, len(video)):
                frame = video[i]
                mask = self.track_object(frame, com)
                com = self.get_center_of_mass(mask)
                object_position = self.map_to_robot_frame(com, K, resolution, horizon)
                object_positions_video.append(object_position)

            object_positions.append(object_positions_video)

        return object_positions


    def segment_object(self, frame):

        segment_video = semantic_segmentation()
        segment_video.load_pascalvoc_model("/home/siri/Projects/videos/")
        segment_video.process_video_pascalvoc("/home/siri/Projects/videos/", overlay = True, frames_per_second= 15, output_video_name="segmented.mp4")

        return mask


    def get_center_of_mass(self, mask):
        
        pass

        return com


    def track_object(self, frame, point_or_mask):
        # Run track anything to get segmentation mask for object in each frame
        pass

        return mask 


    def map_to_robot_frame(self, position, K, resolution, horizon):
        # Combine with depth information and map object position to robot frame using the calibration matrix K

        pass

        return obj_pos # in robot frame
    
    def step(self, action):
