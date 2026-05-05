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

from tracker.config import (
    GRID_COLS, GRID_ROWS, ROBOT_TAGS, ITEM_TAGS,
    DETECTION_ZONE_RATIO, FISHEYE_EDGE_SHRINK,
)
from tracker.field import pixel_to_field, pixel_to_field_continuous
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
    # Strategy: nearest-cell-centre assignment.
    # The tag is assigned to the cell whose centre is closest in grid space.
    # This guarantees exactly one cell is always active — no dead zones —
    # while still requiring the robot to cross the mid-line between cells
    # before the assignment switches (equivalent to Voronoi regions).

    for det in detections:
        tid = det.tag_id
        gx, gy = pixel_to_field_continuous(det.center[0], det.center[1], H)

        # Each cell centre sits at (c + 0.5, r + 0.5) in grid space.
        # Find the nearest one by rounding the shifted coordinate.
        col = int(np.clip(round(gx - 0.5), 0, GRID_COLS - 1))
        row = int(np.clip(round(gy - 0.5), 0, GRID_ROWS - 1))

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

def _cell_corners_px(col, row, H_inv, ratio=1.0):
    """Map one grid cell (col, row) → 4 pixel-space corners via inverse H.

    ratio < 1.0 shrinks the rectangle symmetrically toward the cell centre,
    which is used to draw the inner detection hot-zone.

    Row 0 = top of image, Row GRID_ROWS-1 = bottom.
    Grid-space rectangle for (col, row):
        x ∈ [col, col+1]
        y ∈ [row, row+1]
    """
    pad = (1.0 - ratio) / 2.0
    gy_top = row       + pad
    gy_bot = row + 1.0 - pad
    gx_l   = col       + pad
    gx_r   = col + 1.0 - pad

    pts = np.array([
        [[gx_l, gy_top]],
        [[gx_r, gy_top]],
        [[gx_r, gy_bot]],
        [[gx_l, gy_bot]],
    ], dtype=np.float32)

    return cv2.perspectiveTransform(pts, H_inv).reshape(4, 2).astype(int)


def draw_grid(frame, matrix, H):
    """Draw coloured bounding boxes for every cell onto *frame* (in-place blend).

    Each cell gets:
    • A full-cell border (thin) showing the grid layout.
    • An inner hot-zone box (thick, filled when occupied) that reflects
      DETECTION_ZONE_RATIO.  Cells farther from the image centre receive
      an additional shrink (FISHEYE_EDGE_SHRINK) to compensate for the
      fisheye lens making edge objects appear larger than they are.
    """
    H_inv = np.linalg.inv(H)
    overlay = frame.copy()

    fh, fw = frame.shape[:2]
    img_cx, img_cy = fw / 2.0, fh / 2.0
    # maximum possible distance from image centre to a corner
    max_dist = np.hypot(img_cx, img_cy)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            val  = matrix[r][c]

            # ── full-cell outline ────────────────────────────────────────
            full_corn = _cell_corners_px(c, r, H_inv, ratio=1.0)
            full_pts  = full_corn.reshape((-1, 1, 2))
            cv2.polylines(overlay, [full_pts], True, _CLR_GRID, 1)

            # ── fisheye-compensated hot-zone ratio ───────────────────────
            cx_px, cy_px = full_corn.mean(axis=0)
            radial = np.hypot(cx_px - img_cx, cy_px - img_cy) / max_dist
            zone_ratio = DETECTION_ZONE_RATIO * (1.0 - FISHEYE_EDGE_SHRINK * radial)
            zone_ratio = float(np.clip(zone_ratio, 0.15, 1.0))

            # ── inner hot-zone box ───────────────────────────────────────
            zone_corn = _cell_corners_px(c, r, H_inv, ratio=zone_ratio)
            zone_pts  = zone_corn.reshape((-1, 1, 2))

            fill = _FILL.get(val)
            if fill is not None:
                cv2.fillPoly(overlay, [zone_pts], fill)

            thickness = 2 if val != CELL_EMPTY else 1
            cv2.polylines(overlay, [zone_pts], True, _BORDER[val], thickness)

            # ── coordinate + value label at cell centre ──────────────────
            lcx, lcy = full_corn.mean(axis=0).astype(int)
            cv2.putText(overlay, f"({c},{r})",
                        (lcx - 18, lcy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, _LABEL.get(val, str(val)),
                        (lcx - 10, lcy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
