"""
Robot AprilTag Tracking System
================================
Overhead fixed camera + AprilTags on field + one moving tag on robot.

Requirements:
    pip install pupil-apriltags opencv-python numpy

Usage:
    python main.py
    Press 'q' to quit
"""

import cv2
import numpy as np
from pupil_apriltags import Detector
import time
import threading

# ─────────────────────────────────────────────
# CONFIGURATION  — edit these to match your setup
# ─────────────────────────────────────────────

# ── Source mode ────────────────────────────────────────────────────────────────
RTSP_URL     = "rtsp://admin:admin@192.168.0.172:8554/Streaming/Channels/101"
TEST_VIDEO   = None       # e.g. "field.mp4"
CAMERA_INDEX = 1          # USB camera fallback

# ── Camera / stream settings ──────────────────────────────────────────────────
FRAME_W      = 1280
FRAME_H      = 720
TARGET_FPS   = 1
BRIGHTNESS   = 120
CONTRAST     = 1.3

# ── Display settings ──────────────────────────────────────────────────────────
DISPLAY_W    = 960
DISPLAY_H    = 540

# ── Fisheye undistortion ──────────────────────────────────────────────────────
# Set UNDISTORT_FISHEYE = True to correct barrel / fisheye distortion.
# Adjust FISHEYE_K  (radial strength — larger = stronger correction)
# and    FISHEYE_BALANCE (0 = crop all black edges, 1 = keep entire image).
# You can also tune live with '+'/'-' (K) and '['/']' (balance).
UNDISTORT_FISHEYE  = True
FISHEYE_K          = -0.35        # k1 coefficient (negative = barrel correction)
FISHEYE_BALANCE    = 0.75          # 0.0 – 1.0
FISHEYE_K_STEP     = 0.05         # step per keypress
FISHEYE_BAL_STEP   = 0.05

# ── RTSP-specific options ─────────────────────────────────────────────────────
RTSP_RECONNECT_DELAY = 3
RTSP_TRANSPORT       = "tcp"
RTSP_SOCKET_TIMEOUT  = 10_000_000   # microseconds (10 s) — increase on slow devices

TAG_FAMILY   = "tag36h11"

# ── Grid dimensions ───────────────────────────────────────────────────────────
GRID_COLS = 8
GRID_ROWS = 4

# ── Field quadrilateral (full-resolution frame pixels) ────────────────────────
# Four corners of the playing field — adjust each independently.
# Order: TL, TR, BR, BL  (top-left, top-right, bottom-right, bottom-left)
# Adjust live while running with keys (see HUD in bottom-left of window).
FIELD_QUAD = [[260, 150], [2110, 220], [2080, 1140], [220, 1080]]
QUAD_STEP = 10   # pixels per keypress

DOCKING_TAGS = {100: "docking1", 101: "docking2"}
ROBOT_TAGS   = {577: "Robot 1", 579: "Robot 2"}
ITEM_TAGS    = {580: "Item 1", 581: "Item 2", 582: "Item 3", 583: "Item 4",
                585: "Item 5", 586: "Item 6", 584: "Item 7", 576: "Item 8"}

CORNER_NAMES = ["TL", "TR", "BR", "BL"]


# ─────────────────────────────────────────────
# FIELD MAPPING
# ─────────────────────────────────────────────

def quad_homography(quad):
    """Compute perspective transform from quad corners → grid rectangle."""
    src = np.array(quad, dtype=np.float32)
    dst = np.array([
        [0,         0        ],
        [GRID_COLS, 0        ],
        [GRID_COLS, GRID_ROWS],
        [0,         GRID_ROWS],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def pixel_to_field(px, py, H):
    """Map full-res pixel coords to grid (col, row) using perspective transform H."""
    pt = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), H)[0][0]
    col = int(round(max(0, min(GRID_COLS, pt[0]))))
    row = int(round(max(0, min(GRID_ROWS, pt[1]))))
    return col, row


def adjust_quad(quad, selected, key):
    """
    Move the selected corner with w/a/s/d.
    Returns (updated_quad, changed) where changed=True when something moved.
    """
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


# ─────────────────────────────────────────────
# FISHEYE UNDISTORTION
# ─────────────────────────────────────────────

def build_undistort_maps(frame_w, frame_h, k1, balance):
    """
    Pre-compute the undistortion + rectification maps for a fisheye lens.

    Parameters
    ----------
    frame_w, frame_h : int
        Resolution of the incoming frames.
    k1 : float
        Radial distortion coefficient (negative corrects barrel distortion).
    balance : float  (0.0 – 1.0)
        0 = crop all black borders, 1 = keep the full undistorted image.

    Returns
    -------
    map1, map2 : ndarray
        Remap look-up tables usable with cv2.remap().
    """
    # Approximate camera matrix (principal point at centre, focal ≈ width)
    fx = fy = frame_w
    cx, cy  = frame_w / 2.0, frame_h / 2.0
    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]], dtype=np.float64)

    D = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64)   # k1, k2, k3, k4

    dim = (frame_w, frame_h)

    # New camera matrix that controls how much of the corrected image is shown
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


# ─────────────────────────────────────────────
# VISUALISATION OVERLAY
# ─────────────────────────────────────────────

def draw_overlay(frame, detections, quad, H, selected):
    overlay = frame.copy()

    # Draw field quadrilateral
    pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

    # Draw corner handles — highlight selected
    for i, (cx, cy) in enumerate(quad):
        is_sel = (i == selected)
        dot_color  = (0, 255, 255) if is_sel else (255, 255, 0)
        text_color = (0, 255, 255) if is_sel else (200, 200, 50)
        cv2.circle(overlay, (int(cx), int(cy)), 8 if is_sel else 5, dot_color, -1)
        cv2.putText(overlay, f"{i+1}:{CORNER_NAMES[i]}", (int(cx) + 10, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    for det in detections:
        tid  = det.tag_id
        ctr  = det.center.astype(int)
        corn = det.corners.astype(int)

        if tid in DOCKING_TAGS:
            color = (255, 140, 0)
            label = DOCKING_TAGS[tid]
        elif tid in ROBOT_TAGS:
            color = (0, 180, 255)
            label = ROBOT_TAGS[tid]
        elif tid in ITEM_TAGS:
            color = (0, 210, 130)
            label = ITEM_TAGS[tid]
        else:
            color = (200, 200, 200)
            label = f"ID {tid}"

        cv2.polylines(overlay, [corn.reshape(-1, 1, 2)], True, color, 2)

        col, row = pixel_to_field(det.center[0], det.center[1], H)
        label = f"{label} [{col},{row}]"

        cv2.putText(overlay, label, (ctr[0] - 20, ctr[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    alpha = 0.85
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_quad_hud(display, selected, fisheye_k=None, fisheye_bal=None):
    """Draw corner-selection + move key reference (+ fisheye info)."""
    lines = [
        "Field quad adjust:",
        "1/2/3/4  select corner",
        "  1=TL  2=TR",
        "  4=BL  3=BR",
        "w/s  move up/down",
        "a/d  move left/right",
        f"Active: {selected+1} {CORNER_NAMES[selected]}",
    ]
    if fisheye_k is not None:
        lines.append("")
        lines.append("Fisheye correction:")
        lines.append(f"  +/-   K = {fisheye_k:+.2f}")
        lines.append(f"  [/]   balance = {fisheye_bal:.2f}")
    x = 10
    y = DISPLAY_H - 10 - len(lines) * 18
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i == len(lines) - 1 else (200, 200, 50)
        cv2.putText(display, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        y += 18


def log_positions(detections, H, timestamp_ms=None):
    """Clear terminal and reprint a position table for all detected tags."""
    print("\033[2J\033[H", end="")

    ts = f"  t = {timestamp_ms:.0f} ms" if timestamp_ms is not None else ""
    print(f"── Detected Tags {ts}")
    print(f"  {'ID':>4}  {'Name':<16}  {'Type':<8}  {'Pixel (x,y)':>18}  {'Grid (col,row)':>14}")
    print("  " + "─" * 70)

    for det in sorted(detections, key=lambda d: d.tag_id):
        tid = det.tag_id
        px, py = det.center

        if tid in DOCKING_TAGS:
            name, tag_type = DOCKING_TAGS[tid], "docking"
        elif tid in ROBOT_TAGS:
            name, tag_type = ROBOT_TAGS[tid], "robot"
        elif tid in ITEM_TAGS:
            name, tag_type = ITEM_TAGS[tid], "item"
        else:
            name, tag_type = f"Tag {tid}", "?"

        col, row = pixel_to_field(px, py, H)
        grid_str = f"({col}, {row})"

        print(f"  {tid:>4}  {name:<16}  {tag_type:<8}  ({px:6.0f}, {py:6.0f})  {grid_str:>14}")

    print("─" * 72)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def apply_contrast(frame, contrast):
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=0)


def process_frame(frame, detector, quad, H, selected, undistort_maps=None):
    """Detect tags, return (vis, detections).  Optionally undistort first."""
    if undistort_maps is not None:
        frame = undistort_frame(frame, *undistort_maps)
    frame = apply_contrast(frame, CONTRAST)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    vis = draw_overlay(frame, detections, quad, H, selected)
    return vis, detections


# ─────────────────────────────────────────────
# THREADED VIDEO CAPTURE  — eliminates RTSP lag
# ─────────────────────────────────────────────

class ThreadedVideoCapture:
    """
    Continuously grab frames from a VideoCapture in a background thread,
    keeping only the **latest** frame.  This prevents the RTSP internal
    buffer from filling up and causing a multi-second delay.
    """

    def __init__(self, cap):
        self._cap   = cap
        self._lock  = threading.Lock()
        self._frame = None
        self._ret   = False
        self._running = True
        self._thread  = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    # ── background loop: drain the capture buffer ──────────────────────────
    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret   = ret
                self._frame = frame
            if not ret:
                # avoid busy-spinning when the stream is down
                time.sleep(0.01)

    # ── public API (matches cv2.VideoCapture) ─────────────────────────────
    def read(self):
        """Return the most recent frame (never blocks on decode)."""
        with self._lock:
            return self._ret, self._frame

    def isOpened(self):
        return self._cap.isOpened()

    def get(self, prop_id):
        return self._cap.get(prop_id)

    def set(self, prop_id, value):
        return self._cap.set(prop_id, value)

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()


def open_rtsp(url):
    """
    Open an RTSP stream with TCP transport.
    Waits for the connection to actually produce a frame before
    handing off to the threaded reader — prevents timeout on slow CPUs.
    """
    import os
    opts = []
    if RTSP_TRANSPORT == "tcp":
        opts.append("rtsp_transport;tcp")
    opts.append(f"stimeout;{RTSP_SOCKET_TIMEOUT}")   # socket-level timeout
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("  ⚠  VideoCapture failed to open — returning raw handle.")
        return ThreadedVideoCapture(cap)

    # Warm-up: wait until the first frame actually arrives so the RTSP
    # handshake completes on the main thread before the reader loop starts.
    print("  Waiting for first RTSP frame (warm-up) ...")
    warmup_deadline = time.time() + RTSP_SOCKET_TIMEOUT / 1_000_000
    while time.time() < warmup_deadline:
        ret, frame = cap.read()
        if ret and frame is not None:
            print("  First frame received — starting threaded reader.")
            # Seed the threaded wrapper so .read() returns this frame immediately
            wrapper = ThreadedVideoCapture(cap)
            with wrapper._lock:
                wrapper._ret   = True
                wrapper._frame = frame
            return wrapper
        time.sleep(0.1)

    print("  ⚠  Warm-up timed out — starting threaded reader anyway.")
    return ThreadedVideoCapture(cap)


def show_waiting(window_name):
    placeholder = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for stream...", (20, DISPLAY_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H)
    cv2.imshow(window_name, placeholder)


def main():
    detector = Detector(
        families=TAG_FAMILY,
        nthreads=2,              # match low-core CPUs (2-core Intel)
        quad_decimate=2.0,       # faster detection (trades slight accuracy)
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    # ── Video file mode ──
    if TEST_VIDEO is not None and RTSP_URL is None:
        cap = cv2.VideoCapture(TEST_VIDEO)
        if not cap.isOpened():
            print(f"Error: could not open video '{TEST_VIDEO}'")
            return

        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"Playing: {TEST_VIDEO}  ({fps_video:.0f} fps)  —  press 'q' to quit")

        quad     = [list(c) for c in FIELD_QUAD]
        H        = quad_homography(quad)
        selected = 0
        last_log_ms  = -9999
        LOG_INTERVAL = 500

        # Fisheye undistortion maps (precomputed once, rebuilt on param change)
        fisheye_k   = FISHEYE_K
        fisheye_bal = FISHEYE_BALANCE
        undistort_maps = None
        if UNDISTORT_FISHEYE:
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or FRAME_W
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_H
            undistort_maps = build_undistort_maps(src_w, src_h, fisheye_k, fisheye_bal)
            print(f"Fisheye undistortion ON  K={fisheye_k:.2f}  balance={fisheye_bal:.2f}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            vis, detections = process_frame(frame, detector, quad, H, selected,
                                            undistort_maps)

            if pos_ms - last_log_ms >= LOG_INTERVAL:
                log_positions(detections, H, timestamp_ms=pos_ms)
                last_log_ms = pos_ms

            display = cv2.resize(vis, (DISPLAY_W, DISPLAY_H))
            draw_quad_hud(display, selected,
                          fisheye_k if UNDISTORT_FISHEYE else None,
                          fisheye_bal if UNDISTORT_FISHEYE else None)
            cv2.namedWindow("Robot Tracker — Video", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Robot Tracker — Video", DISPLAY_W, DISPLAY_H)
            cv2.imshow("Robot Tracker — Video", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in (ord('1'), ord('2'), ord('3'), ord('4')):
                selected = key - ord('1')
            elif UNDISTORT_FISHEYE and key in (ord('+'), ord('=')):
                fisheye_k += FISHEYE_K_STEP
                src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or FRAME_W
                src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_H
                undistort_maps = build_undistort_maps(src_w, src_h, fisheye_k, fisheye_bal)
                print(f"Fisheye K = {fisheye_k:+.2f}")
            elif UNDISTORT_FISHEYE and key in (ord('-'), ord('_')):
                fisheye_k -= FISHEYE_K_STEP
                src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or FRAME_W
                src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_H
                undistort_maps = build_undistort_maps(src_w, src_h, fisheye_k, fisheye_bal)
                print(f"Fisheye K = {fisheye_k:+.2f}")
            elif UNDISTORT_FISHEYE and key == ord(']'):
                fisheye_bal = min(1.0, fisheye_bal + FISHEYE_BAL_STEP)
                src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or FRAME_W
                src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_H
                undistort_maps = build_undistort_maps(src_w, src_h, fisheye_k, fisheye_bal)
                print(f"Fisheye balance = {fisheye_bal:.2f}")
            elif UNDISTORT_FISHEYE and key == ord('['):
                fisheye_bal = max(0.0, fisheye_bal - FISHEYE_BAL_STEP)
                src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or FRAME_W
                src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_H
                undistort_maps = build_undistort_maps(src_w, src_h, fisheye_k, fisheye_bal)
                print(f"Fisheye balance = {fisheye_bal:.2f}")
            else:
                quad, changed = adjust_quad(quad, selected, key)
                if changed:
                    H = quad_homography(quad)

        cap.release()
        cv2.destroyAllWindows()
        return

    # ── RTSP stream mode ──
    if RTSP_URL is not None:
        print(f"Connecting to RTSP stream: {RTSP_URL}")
        cap = open_rtsp(RTSP_URL)          # already threaded
        source_label = "Robot Tracker — RTSP"
        print("RTSP stream opened (threaded reader) — press 'q' to quit.")
    else:
        raw_cap = cv2.VideoCapture(CAMERA_INDEX)
        raw_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        raw_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        raw_cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
        raw_cap.set(cv2.CAP_PROP_BRIGHTNESS,   BRIGHTNESS)
        cap = ThreadedVideoCapture(raw_cap)   # wrap USB cam too
        source_label = "Robot Tracker"
        print("Starting USB camera (threaded reader) — press 'q' to quit.")

    quad                 = [list(c) for c in FIELD_QUAD]
    H                    = quad_homography(quad)
    selected             = 0   # which corner is active: 0=TL 1=TR 2=BR 3=BL
    prev_time            = time.time()
    last_log_t           = prev_time
    LOG_INTERVAL         = 0.5
    detections           = []
    consecutive_failures = 0
    first_frame          = True

    # Fisheye undistortion state
    fisheye_k        = FISHEYE_K
    fisheye_bal      = FISHEYE_BALANCE
    undistort_maps   = None
    maps_built       = False      # build maps lazily after first frame arrives
    prev_frame_id    = None       # used to skip duplicate frames from the reader

    while True:
        ret, frame = cap.read()

        # Skip if the threaded reader returned the exact same object (no new frame yet)
        if ret and frame is not None and frame is prev_frame_id:
            time.sleep(0.005)   # yield CPU briefly
            # still process key events so UI stays responsive
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue
        if ret and frame is not None:
            prev_frame_id = frame

        if not ret:
            consecutive_failures += 1
            show_waiting(source_label)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if RTSP_URL is not None and consecutive_failures < 10:
                time.sleep(0.05)
                continue

            if RTSP_URL is not None:
                print(f"\nRTSP stream lost — reconnecting in {RTSP_RECONNECT_DELAY}s ...")
                cap.release()
                time.sleep(RTSP_RECONNECT_DELAY)
                cap = open_rtsp(RTSP_URL)   # new threaded wrapper
                maps_built = False          # rebuild fisheye maps for new stream
                consecutive_failures = 0
                if not cap.isOpened():
                    print("Reconnect failed. Retrying ...")
                else:
                    print("Reconnected (threaded reader).")
            else:
                print("Camera read failed.")
                break
            continue

        if first_frame:
            print("First frame received — stream is live.")
            first_frame = False

        # Build fisheye maps once we know the actual frame size
        if UNDISTORT_FISHEYE and not maps_built:
            src_h, src_w = frame.shape[:2]
            undistort_maps = build_undistort_maps(src_w, src_h, fisheye_k, fisheye_bal)
            maps_built = True
            print(f"Fisheye undistortion ON  K={fisheye_k:.2f}  balance={fisheye_bal:.2f}")

        consecutive_failures = 0
        now = time.time()
        dt  = now - prev_time
        prev_time = now

        vis, detections = process_frame(frame, detector, quad, H, selected,
                                         undistort_maps)

        if now - last_log_t >= LOG_INTERVAL:
            log_positions(detections, H)
            last_log_t = now

        fps = 1.0 / dt if dt > 0 else 0

        display = cv2.resize(vis, (DISPLAY_W, DISPLAY_H))
        cv2.putText(display, f"{fps:.0f} fps", (DISPLAY_W - 80, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        draw_quad_hud(display, selected,
                      fisheye_k if UNDISTORT_FISHEYE else None,
                      fisheye_bal if UNDISTORT_FISHEYE else None)
        cv2.namedWindow(source_label, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(source_label, DISPLAY_W, DISPLAY_H)
        cv2.imshow(source_label, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in (ord('1'), ord('2'), ord('3'), ord('4')):
            selected = key - ord('1')
        elif UNDISTORT_FISHEYE and key in (ord('+'), ord('=')):
            fisheye_k += FISHEYE_K_STEP
            src_h, src_w = frame.shape[:2]
            undistort_maps = build_undistort_maps(src_w, src_h, fisheye_k, fisheye_bal)
            print(f"Fisheye K = {fisheye_k:+.2f}")
        elif UNDISTORT_FISHEYE and key in (ord('-'), ord('_')):
            fisheye_k -= FISHEYE_K_STEP
            src_h, src_w = frame.shape[:2]
            undistort_maps = build_undistort_maps(src_w, src_h, fisheye_k, fisheye_bal)
            print(f"Fisheye K = {fisheye_k:+.2f}")
        elif UNDISTORT_FISHEYE and key == ord(']'):
            fisheye_bal = min(1.0, fisheye_bal + FISHEYE_BAL_STEP)
            src_h, src_w = frame.shape[:2]
            undistort_maps = build_undistort_maps(src_w, src_h, fisheye_k, fisheye_bal)
            print(f"Fisheye balance = {fisheye_bal:.2f}")
        elif UNDISTORT_FISHEYE and key == ord('['):
            fisheye_bal = max(0.0, fisheye_bal - FISHEYE_BAL_STEP)
            src_h, src_w = frame.shape[:2]
            undistort_maps = build_undistort_maps(src_w, src_h, fisheye_k, fisheye_bal)
            print(f"Fisheye balance = {fisheye_bal:.2f}")
        else:
            quad, changed = adjust_quad(quad, selected, key)
            if changed:
                H = quad_homography(quad)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal quad config:")
    print(f"FIELD_QUAD = {quad}")
    if UNDISTORT_FISHEYE:
        print(f"Final fisheye  K = {fisheye_k:+.2f}   balance = {fisheye_bal:.2f}")


if __name__ == "__main__":
    main()
