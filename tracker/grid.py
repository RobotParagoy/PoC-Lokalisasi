"""Grid occupancy matrix — build, JSON-log, and draw the 8×4 platform grid.

Each cell is 30 × 30 cm.  Values:
    0  — empty / unoccupied lane (no AprilTag)
    1  — robot
    2  — goods (item)
    4  — docking area  (hard-coded at columns 1-2, row 0)

Coordinate convention  (col, row):
      col →  0  1  2  3  4  5  6  7
    row 0
    row 1        ← docking (col 0)
    row 2        ← docking (col 0)
    row 3
"""

import json
import cv2
import numpy as np

from tracker.config import GRID_COLS, GRID_ROWS, ROBOT_TAGS, ITEM_TAGS
from tracker.field import pixel_to_field
from tracker.tags import classify_tag, tag_orientation

# ── Cell value constants ─────────────────────────────────────────────────────
CELL_EMPTY   = 0
CELL_ROBOT   = 1
CELL_GOODS   = 2
CELL_DOCKING = 4

# Fixed docking positions (col, row)
DOCKING_CELLS = [(0, 1), (0, 2)]

# ── Colours (BGR) ────────────────────────────────────────────────────────────
_CLR_EMPTY   = (120, 120, 120)
_CLR_ROBOT   = (0, 180, 255)     # orange-ish
_CLR_GOODS   = (0, 210, 130)     # green
_CLR_DOCKING = (255, 140, 0)     # blue-ish
_CLR_GRID    = (90, 90, 90)

_FILL = {
    CELL_EMPTY:   None,
    CELL_ROBOT:   (0, 180, 255),
    CELL_GOODS:   (0, 210, 130),
    CELL_DOCKING: (255, 140, 0),
}

_BORDER = {
    CELL_EMPTY:   _CLR_GRID,
    CELL_ROBOT:   _CLR_ROBOT,
    CELL_GOODS:   _CLR_GOODS,
    CELL_DOCKING: _CLR_DOCKING,
}

_LABEL = {
    CELL_EMPTY:   "0",
    CELL_ROBOT:   "1 R",
    CELL_GOODS:   "2 G",
    CELL_DOCKING: "4 D",
}


# ═════════════════════════════════════════════════════════════════════════════
# Matrix construction
# ═════════════════════════════════════════════════════════════════════════════

def build_grid(detections, H):
    """Return (matrix, coord_dict) for the current frame.

    matrix   — list[row][col]   (row-major, 4 × 8)
    coord_dict — { "(col,row)": value, … }  ready for JSON serialisation
      value is 0 for empty, 4 for docking placeholder,
      or [tag_id, angle_deg] for cells with a detected AprilTag.
    """
    matrix = [[CELL_EMPTY] * GRID_COLS for _ in range(GRID_ROWS)]
    cell_det = {}  # (col, row) -> detection object

    # permanent docking cells
    for c, r in DOCKING_CELLS:
        matrix[r][c] = CELL_DOCKING

    # populate from detections
    for det in detections:
        tid = det.tag_id
        col, row = pixel_to_field(det.center[0], det.center[1], H)

        if tid in ROBOT_TAGS:
            matrix[row][col] = CELL_ROBOT
            cell_det[(col, row)] = det
        elif tid in ITEM_TAGS:
            matrix[row][col] = CELL_GOODS
            cell_det[(col, row)] = det
        # docking tags keep their CELL_DOCKING value

    coord_dict = {}
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            det = cell_det.get((c, r))
            if det is not None:
                angle = tag_orientation(det.corners)
                coord_dict[f"({c},{r})"] = [det.tag_id, angle]
            else:
                coord_dict[f"({c},{r})"] = matrix[r][c]

    return matrix, coord_dict


# ═════════════════════════════════════════════════════════════════════════════
# JSON logging
# ═════════════════════════════════════════════════════════════════════════════

def grid_to_json(coord_dict):
    """Serialise coordinate dict → compact JSON string."""
    return json.dumps(coord_dict, indent=2)


def log_grid(matrix, coord_dict):
    """Print the grid matrix as JSON + a small ASCII visualisation."""
    print("\n── Platform Grid (JSON) ─────────────────────")
    print(grid_to_json(coord_dict))

    # quick ASCII table (row 3 at top, row 0 at bottom — matches camera view)
    print("\n── Grid Visual ──")
    header = "       " + "  ".join(f" {c} " for c in range(GRID_COLS))
    print(header)
    for r in range(GRID_ROWS):
        cells = "  ".join(f"[{matrix[r][c]}]" for c in range(GRID_COLS))
        print(f"  r{r}   {cells}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═════════════════════════════════════════════════════════════════════════════

def _cell_corners_px(col, row, H_inv):
    """Map one grid cell (col, row) → 4 pixel-space corners via inverse H.

    Row 0 = top of image, Row GRID_ROWS-1 = bottom.
    Grid-space rectangle for (col, row):
        x ∈ [col, col+1]
        y ∈ [row, row+1]
    """
    gy_top = row
    gy_bot = row + 1
    gx_l   = col
    gx_r   = col + 1

    pts = np.array([
        [[gx_l, gy_top]],
        [[gx_r, gy_top]],
        [[gx_r, gy_bot]],
        [[gx_l, gy_bot]],
    ], dtype=np.float32)

    return cv2.perspectiveTransform(pts, H_inv).reshape(4, 2).astype(int)


def draw_grid(frame, matrix, H):
    """Draw coloured bounding boxes for every cell onto *frame* (in-place blend)."""
    H_inv = np.linalg.inv(H)
    overlay = frame.copy()

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            val  = matrix[r][c]
            corn = _cell_corners_px(c, r, H_inv)
            pts  = corn.reshape((-1, 1, 2))

            # translucent fill for non-empty cells
            fill = _FILL.get(val)
            if fill is not None:
                cv2.fillPoly(overlay, [pts], fill)

            # cell border
            thickness = 2 if val != CELL_EMPTY else 1
            cv2.polylines(overlay, [pts], True, _BORDER[val], thickness)

            # coordinate + value label at cell centre
            cx, cy = corn.mean(axis=0).astype(int)
            cv2.putText(overlay, f"({c},{r})",
                        (cx - 18, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, _LABEL.get(val, str(val)),
                        (cx - 10, cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
