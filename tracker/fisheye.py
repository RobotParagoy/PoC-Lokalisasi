"""Fisheye lens undistortion utilities."""

import cv2
import numpy as np


def build_undistort_maps(frame_w, frame_h, k1, balance):
    """Pre-compute undistortion + rectification maps for a fisheye lens."""
    fx = fy = frame_w
    cx, cy  = frame_w / 2.0, frame_h / 2.0
    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]], dtype=np.float64)

    D = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64)
    dim = (frame_w, frame_h)

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, dim, np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, dim, cv2.CV_16SC2
    )
    return map1, map2


def undistort_frame(frame, map1, map2):
    """Apply precomputed fisheye undistortion maps to a frame."""
    return cv2.remap(frame, map1, map2,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)
