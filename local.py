import cv2
import numpy as np
from pupil_apriltags import Detector
import time
import os
import threading

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RTSP_URL     = "rtsp://admin:admin@192.168.0.172:8554/Streaming/Channels/102"
CAMERA_INDEX = 0  # Used if RTSP_URL is None

DISPLAY_W    = 960
DISPLAY_H    = 540

GRID_COLS = 8
GRID_ROWS = 4

SCALE = 0.4          # Downscale for faster detection
FRAME_SKIP = 1       # With threading, we can usually set this to 1

CORNER_TAGS = {100: (0, 0), 101: (0, GRID_ROWS), 102: (GRID_COLS, 0), 103: (GRID_COLS, GRID_ROWS)}
ROBOT_TAGS  = {577: "Robot 1", 579: "Robot 2"}
ITEM_TAGS   = {580: "Item 1", 581: "Item 2", 582: "Item 3", 583: "Item 4", 585: "Item 5"}

# ─────────────────────────────────────────────
# THREADED CAPTURE (THE LAG KILLER)
# ─────────────────────────────────────────────
class FreshFrame:
    def __init__(self, source):
        # FFMPEG low-latency options: UDP is faster than TCP for RTSP on LAN
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
            "|max_delay;0|reorder_queue_size;0"
        )

        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret = False
        self.frame = None
        self.stopped = False
        self._lock = threading.Lock()

        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.ret = False
                break
            ret, frame = self.cap.read()
            if ret:
                # Drain any buffered frames — keep only the freshest one
                while True:
                    r, f = self.cap.read()
                    if not r:
                        break
                    ret, frame = r, f
                with self._lock:
                    self.ret, self.frame = ret, frame

    def read(self):
        with self._lock:
            return self.ret, self.frame

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# ─────────────────────────────────────────────
# HOMOGRAPHY & UTILS
# ─────────────────────────────────────────────
def compute_homography(corner_detections):
    src_pts, dst_pts = [], []
    for tag_id, pixel_xy in corner_detections.items():
        if tag_id in CORNER_TAGS:
            src_pts.append(pixel_xy)
            dst_pts.append(CORNER_TAGS[tag_id])
    if len(src_pts) < 4:
        return None
    H, _ = cv2.findHomography(np.array(src_pts), np.array(dst_pts))
    return H

def pixel_to_field(H, px, py):
    pt = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    fx, fy = result[0][0][0], result[0][0][1]
    col, row = int(round(max(0, min(GRID_COLS, fx)))), int(round(max(0, min(GRID_ROWS, fy))))
    return col, row

def draw_overlay(frame, detections, H):
    # Draw directly on frame — skip addWeighted blend, saves a full frame copy+op
    for det in detections:
        tid = det.tag_id
        ctr = det.center.astype(int)

        if tid in CORNER_TAGS: color, label = (180, 100, 255), f"CORNER {tid}"
        elif tid in ROBOT_TAGS: color, label = (0, 180, 255), ROBOT_TAGS[tid]
        elif tid in ITEM_TAGS: color, label = (0, 210, 130), ITEM_TAGS[tid]
        else: color, label = (200, 200, 200), f"ID {tid}"

        cv2.polylines(frame, [det.corners.astype(int).reshape(-1, 1, 2)], True, color, 2)

        if H is not None:
            col, row = pixel_to_field(H, det.center[0], det.center[1])
            label += f" [{col},{row}]"

        cv2.putText(frame, label, (ctr[0] - 20, ctr[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return frame

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    detector = Detector(
        families="tag36h11",
        nthreads=2,       # old hardware: more threads = more context switching overhead
        quad_decimate=3.0, # higher = faster detection, slight accuracy tradeoff
        quad_sigma=0.8,    # mild blur reduces false positives at low res
        decode_sharpening=0.25,
        refine_edges=0,    # disable edge refinement — saves time on slow CPUs
    )

    H = None
    source = RTSP_URL if RTSP_URL else CAMERA_INDEX
    stream = FreshFrame(source)
    
    prev_time = time.time()
    print(f"🚀 Threaded Tracker Started: Streaming from {source}")

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                continue

            # Detection on downscaled image
            small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)

            # Upscale coordinates back
            corner_px = {}
            for det in detections:
                det.center *= (1 / SCALE)
                det.corners *= (1 / SCALE)
                if det.tag_id in CORNER_TAGS:
                    corner_px[det.tag_id] = (det.center[0], det.center[1])

            # Update Homography only if all corners found
            if len(corner_px) == 4:
                H = compute_homography(corner_px)

            # Visualization
            vis = draw_overlay(frame, detections, H)
            
            # FPS Calculation
            now = time.time()
            fps = 1 / (now - prev_time)
            prev_time = now

            display = cv2.resize(vis, (DISPLAY_W, DISPLAY_H))
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Robot Tracker (Low Latency)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                H = None
                print("Homography reset")

    finally:
        stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()