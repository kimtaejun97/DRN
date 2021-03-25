# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

import os
import cv2
import numpy as np
from tqdm import tqdm

from rt_gene.gaze_tools import get_endpoint
from rt_gene.gaze_tools import get_error


class GazeEstimatorBase(object):
    """This class encapsulates a deep neural network for gaze estimation.

    It retrieves two image streams, one containing the left eye and another containing the right eye.
    It synchronizes these two images with the estimated head pose.
    The images are then converted in a suitable format, and a forward pass of the deep neural network
    results in the estimated gaze for this frame. The estimated gaze is then published in the (theta, phi) notation."""
    def __init__(self, device_id_gaze, model_files):
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "8"
        tqdm.write("PyTorch using {} threads.".format(os.environ["OMP_NUM_THREADS"]))
        self.device_id_gazeestimation = device_id_gaze
        self.model_files = model_files

        if not isinstance(model_files, list):
            self.model_files = [model_files]

        if len(self.model_files) == 1:
            self._gaze_offset = 0.11
        else:
            self._gaze_offset = 0.0

    def estimate_gaze_twoeyes(self, inference_input_left_list, inference_input_right_list, inference_headpose_list):
        pass

    def input_from_image(self, cv_image):
        pass

    @staticmethod
    def visualize_eye_result(color_img, est_gaze, center_coor, facebox, people, selection,GTlabel=[]):
        # est_gaze = [theta,phi]
        """Here, we take the original eye eye_image and overlay the estimated gaze."""
        output_image = np.copy(color_img)

        center_x = facebox[0] +center_coor[0]
        center_y = facebox[1]+ center_coor[1]


        endpoint_x, endpoint_y = get_endpoint(est_gaze[0], est_gaze[1], center_x, center_y, 150)

        #flip vector
        # flip_endpoint_x, flip_endpoint_y = get_endpoint(est_gaze[0], -est_gaze[1], center_x, center_y, 150)
        # cv2.arrowedLine(output_image, (int(center_x), int(center_y)), (int(flip_endpoint_x), int(flip_endpoint_y)), (255, 0, 0), 2)
        # print("gaze_phi:",est_gaze[1])
        # print("flip gaze_phi:",-est_gaze[1])

        gaze_error = 0
        if GTlabel != []:
            GT_endpoint_x, GT_endpoint_y = get_endpoint(GTlabel[4],GTlabel[3],center_x, center_y, 150)
            gaze_error = get_error([GT_endpoint_x, GT_endpoint_y], [endpoint_x, endpoint_y], [center_x, center_y])

        #Smoothing
        # endpoint_x, endpoint_y = people.getEndpointAverage(endpoint_x, endpoint_y, selection)

        cv2.arrowedLine(output_image, (int(center_x), int(center_y)), (int(endpoint_x), int(endpoint_y)), (255, 0, 0), 2)

        return output_image, gaze_error
