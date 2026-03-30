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

# ── Source mode ────────────────────────────────────────────────────────────────
# Priority: RTSP_URL  →  TEST_VIDEO  →  CAMERA_INDEX
#   • Set RTSP_URL to an RTSP address to stream from an IP camera / NVR.
#   • Set TEST_VIDEO to a file path to replay a recorded video.
#   • Otherwise the USB camera at CAMERA_INDEX is used.
RTSP_URL     = "rtsp://admin:password@192.168.1.100:554/stream1"  # ← edit this
TEST_VIDEO   = None       # e.g. "field.mp4"
CAMERA_INDEX = 1          # USB camera fallback

# ── Camera / stream settings ──────────────────────────────────────────────────
FRAME_W      = 1280
FRAME_H      = 720
TARGET_FPS   = 60         # request fps (USB cam only; RTSP uses source fps)
BRIGHTNESS   = 120        # camera brightness (USB cam only)
CONTRAST     = 1.3        # contrast multiplier applied in software

# ── Display settings ──────────────────────────────────────────────────────────
# The display window is resized to these dimensions so even a 4K RTSP stream
# fits comfortably on screen.  Detection still runs on the full-res frame.
DISPLAY_W    = 960
DISPLAY_H    = 540

# ── RTSP-specific options ─────────────────────────────────────────────────────
RTSP_RECONNECT_DELAY = 3  # seconds to wait before reconnecting on drop
RTSP_TRANSPORT       = "tcp"  # "tcp" is more reliable than default "udp"

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

def apply_contrast(frame, contrast):
    """Scale pixel values around mid-grey by the given multiplier."""
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=0)


def process_frame(frame, detector, H, dt):
    """Detect tags, update homography, return (vis, H, detections)."""
    frame = apply_contrast(frame, CONTRAST)
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


def open_rtsp(url):
    """Open an RTSP stream with TCP transport for reliability."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if RTSP_TRANSPORT == "tcp":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise latency
        # Force TCP — avoids UDP packet-loss artefacts
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    return cap


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
    if TEST_VIDEO is not None and RTSP_URL is None:
        cap = cv2.VideoCapture(TEST_VIDEO)
        if not cap.isOpened():
            print(f"Error: could not open video '{TEST_VIDEO}'")
            return

        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"Playing: {TEST_VIDEO}  ({fps_video:.0f} fps)  —  press 'q' to quit")

        last_log_ms   = -9999
        LOG_INTERVAL  = 500
        detections    = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            vis, H, detections = process_frame(frame, detector, H, dt=1/fps_video)

            if pos_ms - last_log_ms >= LOG_INTERVAL:
                log_positions(detections, H, timestamp_ms=pos_ms)
                last_log_ms = pos_ms

            display = cv2.resize(vis, (DISPLAY_W, DISPLAY_H))
            cv2.namedWindow("Robot Tracker — Video", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Robot Tracker — Video", DISPLAY_W, DISPLAY_H)
            cv2.imshow("Robot Tracker — Video", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return

    # ── RTSP stream mode (real-time IP camera) ──
    if RTSP_URL is not None:
        print(f"Connecting to RTSP stream: {RTSP_URL}")
        cap = open_rtsp(RTSP_URL)
        if not cap.isOpened():
            print("Error: could not open RTSP stream.")
            return
        source_label = "Robot Tracker — RTSP"
        print(f"RTSP stream opened — press 'q' to quit, 'r' to reset homography.")
    else:
        # ── USB camera fallback ──
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
        cap.set(cv2.CAP_PROP_BRIGHTNESS,   BRIGHTNESS)
        source_label = "Robot Tracker"
        print("Starting USB camera — press 'q' to quit, 'r' to reset homography.")

    prev_time    = time.time()
    last_log_t   = prev_time
    LOG_INTERVAL = 0.5
    detections   = []
    consecutive_failures = 0

    while True:
        ret, frame = cap.read()

        # ── Handle frame-read failures (RTSP auto-reconnect) ──
        if not ret:
            consecutive_failures += 1
            if RTSP_URL is not None and consecutive_failures < 10:
                # Tolerate transient drops
                time.sleep(0.05)
                continue

            if RTSP_URL is not None:
                print(f"\n⚠  RTSP stream lost — reconnecting in {RTSP_RECONNECT_DELAY}s …")
                cap.release()
                time.sleep(RTSP_RECONNECT_DELAY)
                cap = open_rtsp(RTSP_URL)
                consecutive_failures = 0
                if not cap.isOpened():
                    print("Reconnect failed. Retrying …")
                else:
                    print("Reconnected.")
                continue
            else:
                print("Camera read failed.")
                break

        consecutive_failures = 0
        now = time.time()
        dt  = now - prev_time
        prev_time = now

        vis, H, detections = process_frame(frame, detector, H, dt)

        if now - last_log_t >= LOG_INTERVAL:
            log_positions(detections, H)
            last_log_t = now

        fps = 1.0 / dt if dt > 0 else 0

        # Resize for display — keeps the window manageable for high-res streams
        display = cv2.resize(vis, (DISPLAY_W, DISPLAY_H))
        cv2.putText(display, f"{fps:.0f} fps", (DISPLAY_W - 80, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.namedWindow(source_label, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(source_label, DISPLAY_W, DISPLAY_H)
        cv2.imshow(source_label, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            H = None
            print("Homography reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()