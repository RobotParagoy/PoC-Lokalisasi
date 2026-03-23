"""
Robot AprilTag Tracking System
================================
Overhead fixed camera + AprilTags on field + one moving tag on robot.

Requirements:
    pip install pupil-apriltags opencv-python numpy

Field layout (from your image):
    - 4 corner tags  : field localization anchors (define coordinate system)
    - Interior tags  : static zone/item markers (ID → zone lookup)
    - 1 moving tag   : robot tag (particle filter tracked)

Usage:
    python robot_tracker.py
    Press 'q' to quit, 'r' to reset particle filter
"""

import cv2
import numpy as np
from pupil_apriltags import Detector
import time

# ─────────────────────────────────────────────
# CONFIGURATION  — edit these to match your setup
# ─────────────────────────────────────────────

CAMERA_INDEX = 0          # USB camera index (try 1 if 0 doesn't work)
FRAME_W      = 1280
FRAME_H      = 720
TARGET_FPS   = 60         # request 60fps from camera

# Set to a video path to test with a video file (e.g. "field.mp4")
# Set to None to use the live camera
TEST_VIDEO   = "field.mp4"

TAG_FAMILY   = "tag36h11" # most robust family — use this for all your tags

# ── Grid dimensions ───────────────────────────────────────────────────────────
# (0,0) = bottom-left   (GRID_COLS, GRID_ROWS) = top-right
GRID_COLS = 8
GRID_ROWS = 4

# Corner tag IDs → grid position (col, row)
# y-axis is FLIPPED vs image pixels: row 0 = image bottom, row 4 = image top
CORNER_TAGS = {
    100: (0,          0),           # bottom-left
    101: (0,          GRID_ROWS),   # top-left
    102: (GRID_COLS,  0),           # bottom-right
    103: (GRID_COLS,  GRID_ROWS),   # top-right
}

# Robot tag IDs  — moving entities on the field
ROBOT_TAGS = {
    577: "Robot 1",
    579: "Robot 2",
}

# Item tag IDs — static objects on the field
ITEM_TAGS = {
    580: "Item 1",
    581: "Item 2",
    582: "Item 3",
    583: "Item 4",
    585: "Item 5",
}



# ─────────────────────────────────────────────
# HOMOGRAPHY  (pixel → field cm)
# ─────────────────────────────────────────────

def compute_homography(corner_detections):
    """
    corner_detections: dict {tag_id: (pixel_x, pixel_y)}
    Returns 3x3 homography matrix H such that field_pt = H @ pixel_pt
    """
    src_pts = []  # pixel coords
    dst_pts = []  # field coords (cm)
    for tag_id, pixel_xy in corner_detections.items():
        if tag_id in CORNER_TAGS:
            src_pts.append(pixel_xy)
            dst_pts.append(CORNER_TAGS[tag_id])
    if len(src_pts) < 4:
        return None
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H

def pixel_to_field(H, px, py):
    """Apply homography to convert pixel (px, py) → field (x_cm, y_cm)."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    return float(result[0][0][0]), float(result[0][0][1])


# ─────────────────────────────────────────────
# VISUALISATION OVERLAY
# ─────────────────────────────────────────────

def draw_overlay(frame, detections, H):
    overlay = frame.copy()

    for det in detections:
        tid  = det.tag_id
        ctr  = det.center.astype(int)
        corn = det.corners.astype(int)

        if tid in CORNER_TAGS:
            color = (180, 100, 255)         # purple — corner anchors
            label = f"CORNER {tid}"
        elif tid in ROBOT_TAGS:
            color = (0, 180, 255)           # amber — robots
            label = ROBOT_TAGS[tid]
        elif tid in ITEM_TAGS:
            color = (0, 210, 130)           # teal — items
            label = ITEM_TAGS[tid]
        else:
            color = (200, 200, 200)         # grey — unknown
            label = f"ID {tid}"

        cv2.polylines(overlay, [corn.reshape(-1, 1, 2)], True, color, 2)

        # Show grid cell in label if homography is available
        if H is not None:
            fx, fy = pixel_to_field(H, det.center[0], det.center[1])
            col, row = field_to_grid(fx, fy)
            label = f"{label} [{col},{row}]"

        cv2.putText(overlay, label, (ctr[0] - 20, ctr[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # HUD
    homography_status = "Homography OK" if H is not None else "Need 4 corner tags"
    cv2.putText(overlay, homography_status, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 220, 80) if H is not None else (0, 80, 220), 1, cv2.LINE_AA)

    alpha = 0.85
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def field_to_grid(fx, fy):
    """Clamp and round continuous field coords to integer grid cell (col, row)."""
    col = int(round(max(0, min(GRID_COLS, fx))))
    row = int(round(max(0, min(GRID_ROWS, fy))))
    return col, row


def log_positions(detections, H, timestamp_ms=None):
    """Clear terminal and reprint a position table for all detected tags."""
    print("\033[2J\033[H", end="")   # clear screen + move cursor to top

    ts = f"  t = {timestamp_ms:.0f} ms" if timestamp_ms is not None else ""
    print(f"── Detected Tags {ts}")
    print(f"  {'ID':>4}  {'Name':<16}  {'Type':<6}  {'Pixel (x,y)':>18}  {'Grid (col,row)':>14}")
    print("  " + "─" * 68)

    for det in sorted(detections, key=lambda d: d.tag_id):
        tid = det.tag_id
        px, py = det.center

        if tid in CORNER_TAGS:
            name, tag_type = f"Corner {tid}", "corner"
        elif tid in ROBOT_TAGS:
            name, tag_type = ROBOT_TAGS[tid], "robot"
        elif tid in ITEM_TAGS:
            name, tag_type = ITEM_TAGS[tid], "item"
        else:
            name, tag_type = f"Tag {tid}", "?"

        if H is not None:
            fx, fy = pixel_to_field(H, px, py)
            col, row = field_to_grid(fx, fy)
            grid_str = f"({col}, {row})"
        else:
            grid_str = "no homography"

        print(f"  {tid:>4}  {name:<16}  {tag_type:<6}  ({px:6.0f}, {py:6.0f})  {grid_str:>14}")

    print("───────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def process_frame(frame, detector, H, dt):
    """Detect tags, update homography, return (vis, H, detections)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    # ── Build / refresh homography from corner tags ──
    corner_px = {}
    for det in detections:
        if det.tag_id in CORNER_TAGS:
            corner_px[det.tag_id] = (det.center[0], det.center[1])
    if len(corner_px) == 4:
        H = compute_homography(corner_px)

    vis = draw_overlay(frame, detections, H)
    return vis, H, detections


def main():
    detector = Detector(
        families=TAG_FAMILY,
        nthreads=4,
        quad_decimate=1.5,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    H = None

    # ── Video file mode ──
    if TEST_VIDEO is not None:
        cap = cv2.VideoCapture(TEST_VIDEO)
        if not cap.isOpened():
            print(f"Error: could not open video '{TEST_VIDEO}'")
            return

        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"Playing: {TEST_VIDEO}  ({fps_video:.0f} fps)  —  press 'q' to quit")

        last_log_ms   = -9999   # tracks when we last printed the table
        LOG_INTERVAL  = 500     # ms between console refreshes
        detections    = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break   # end of video

            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            vis, H, detections = process_frame(frame, detector, H, dt=1/fps_video)

            if pos_ms - last_log_ms >= LOG_INTERVAL:
                log_positions(detections, H, timestamp_ms=pos_ms)
                last_log_ms = pos_ms

            cv2.imshow("Robot Tracker — Video", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return

    # ── Live camera mode ──
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    prev_time    = time.time()
    last_log_t   = prev_time
    LOG_INTERVAL = 0.5          # seconds between console refreshes
    detections   = []
    print("Starting — point camera at field.  Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        now = time.time()
        dt  = now - prev_time
        prev_time = now

        vis, H, detections = process_frame(frame, detector, H, dt)

        if now - last_log_t >= LOG_INTERVAL:
            log_positions(detections, H)
            last_log_t = now

        fps = 1.0 / dt if dt > 0 else 0
        cv2.putText(vis, f"{fps:.0f} fps", (FRAME_W - 75, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow("Robot Tracker", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()