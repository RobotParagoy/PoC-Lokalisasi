"""Field mapping — perspective transform between pixel and grid coordinates."""

import cv2
import numpy as np

from tracker.config import GRID_COLS, GRID_ROWS, QUAD_STEP, CORNER_NAMES


def quad_homography(quad):
    """Compute perspective transform from quad corners to grid rectangle."""
    src = np.array(quad, dtype=np.float32)
    dst = np.array([
        [0,         0        ],
        [GRID_COLS, 0        ],
        [GRID_COLS, GRID_ROWS],
        [0,         GRID_ROWS],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def pixel_to_field(px, py, H):
    """Map full-res pixel coords to grid index (col, row)."""
    pt = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), H)[0][0]
    col = int(max(0, min(GRID_COLS - 1, pt[0])))
    row = GRID_ROWS - 1 - int(max(0, min(GRID_ROWS - 1, pt[1])))
    return col, row


def adjust_quad(quad, selected, key):
    """Move the selected corner with w/a/s/d."""
    dx, dy = 0, 0
    if   key == ord('w'): dy = -QUAD_STEP
    elif key == ord('s'): dy = +QUAD_STEP
    elif key == ord('a'): dx = -QUAD_STEP
    elif key == ord('d'): dx = +QUAD_STEP
    else:
        return quad, False

    new_quad = [list(c) for c in quad]
    new_quad[selected][0] += dx
    new_quad[selected][1] += dy
    print(f"FIELD_QUAD = {new_quad}  # moved {CORNER_NAMES[selected]}")
    return new_quad, True
