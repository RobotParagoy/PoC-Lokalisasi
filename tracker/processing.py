"""Frame processing, state management, and unified key handler."""

import cv2

from tracker.config import (
    CONTRAST, UNDISTORT_FISHEYE, FISHEYE_K, FISHEYE_BALANCE,
    FISHEYE_K_STEP, FISHEYE_BAL_STEP, FIELD_QUAD,
)
from tracker.field import quad_homography, adjust_quad
from tracker.fisheye import build_undistort_maps, undistort_frame
from tracker.overlay import draw_overlay
from tracker.grid import build_grid, draw_grid


def apply_contrast(frame, contrast):
    """Apply contrast scaling to a frame."""
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=0)


def process_frame(frame, detector, quad, H, selected, undistort_maps=None):
    """Detect tags and return (visualised_frame, detections, matrix, coord_dict)."""
    if undistort_maps is not None:
        frame = undistort_frame(frame, *undistort_maps)
    frame = apply_contrast(frame, CONTRAST)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    # build occupancy grid
    matrix, coord_dict = build_grid(detections, H)

    vis = draw_overlay(frame, detections, quad, H, selected)
    vis = draw_grid(vis, matrix, H)
    return vis, detections, matrix, coord_dict


def make_state():
    """Create the shared mutable state dict."""
    return {
        'selected':       0,
        'quad':           [list(c) for c in FIELD_QUAD],
        'H':              None,
        'fisheye_k':      FISHEYE_K,
        'fisheye_bal':    FISHEYE_BALANCE,
        'undistort_maps': None,
        'frame_size':     None,
    }


def handle_keypress(key, state):
    """Handle keyboard input for corner selection, quad adjustment, and fisheye tuning."""
    if key == 255:
        return False

    if key == ord('q'):
        return True

    if key in (ord('1'), ord('2'), ord('3'), ord('4')):
        state['selected'] = key - ord('1')
        return False

    if UNDISTORT_FISHEYE and key in (ord('+'), ord('=')):
        state['fisheye_k'] += FISHEYE_K_STEP
        if state['frame_size'] is not None:
            sw, sh = state['frame_size']
            state['undistort_maps'] = build_undistort_maps(sw, sh, state['fisheye_k'], state['fisheye_bal'])
        print(f"Fisheye K = {state['fisheye_k']:+.2f}")
        return False

    if UNDISTORT_FISHEYE and key in (ord('-'), ord('_')):
        state['fisheye_k'] -= FISHEYE_K_STEP
        if state['frame_size'] is not None:
            sw, sh = state['frame_size']
            state['undistort_maps'] = build_undistort_maps(sw, sh, state['fisheye_k'], state['fisheye_bal'])
        print(f"Fisheye K = {state['fisheye_k']:+.2f}")
        return False

    if UNDISTORT_FISHEYE and key == ord(']'):
        state['fisheye_bal'] = min(1.0, state['fisheye_bal'] + FISHEYE_BAL_STEP)
        if state['frame_size'] is not None:
            sw, sh = state['frame_size']
            state['undistort_maps'] = build_undistort_maps(sw, sh, state['fisheye_k'], state['fisheye_bal'])
        print(f"Fisheye balance = {state['fisheye_bal']:.2f}")
        return False

    if UNDISTORT_FISHEYE and key == ord('['):
        state['fisheye_bal'] = max(0.0, state['fisheye_bal'] - FISHEYE_BAL_STEP)
        if state['frame_size'] is not None:
            sw, sh = state['frame_size']
            state['undistort_maps'] = build_undistort_maps(sw, sh, state['fisheye_k'], state['fisheye_bal'])
        print(f"Fisheye balance = {state['fisheye_bal']:.2f}")
        return False

    quad, changed = adjust_quad(state['quad'], state['selected'], key)
    if changed:
        state['quad'] = quad
        state['H'] = quad_homography(quad)

    return False
