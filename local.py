import cv2
import numpy as np
from pupil_apriltags import Detector
import subprocess
import threading
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RTSP_URL   = "rtsp://admin:admin@192.168.0.172:8554/Streaming/Channels/102"

DISPLAY_W  = 960
DISPLAY_H  = 540
DETECT_W   = 384   # FFmpeg resizes here before handing frame to Python
DETECT_H   = 216

GRID_COLS  = 8
GRID_ROWS  = 4

# Scale factors to map detect-res coords → display-res coords
SX = DISPLAY_W / DETECT_W
SY = DISPLAY_H / DETECT_H

CORNER_TAGS = {100:(0,0), 101:(0,GRID_ROWS), 102:(GRID_COLS,0), 103:(GRID_COLS,GRID_ROWS)}
ROBOT_TAGS  = {577:"Robot 1", 579:"Robot 2"}
ITEM_TAGS   = {580:"Item 1", 581:"Item 2", 582:"Item 3", 583:"Item 4", 585:"Item 5"}


# ─────────────────────────────────────────────
# FFMPEG PIPE STREAM  (lowest possible latency)
# ─────────────────────────────────────────────
class FFmpegStream:
    """
    Spawns ffmpeg as a subprocess and reads raw BGR frames from its stdout.
    FFmpeg handles RTSP with aggressive low-latency flags — no OpenCV buffer.
    The reader thread always keeps only the latest frame.
    """
    def __init__(self, url, w, h):
        self.w, self.h = w, h
        self._frame = None
        self._lock = threading.Lock()
        self._stopped = False

        cmd = [
            "ffmpeg",
            "-loglevel",        "quiet",
            "-fflags",          "nobuffer",         # no input buffer
            "-flags",           "low_delay",         # low-delay mode
            "-strict",          "experimental",
            "-avioflags",       "direct",            # no avio buffer
            "-probesize",       "32",                # minimal stream probe
            "-analyzeduration", "0",                 # skip stream analysis
            "-rtsp_transport",  "udp",               # UDP = lower latency on LAN
            "-i",               url,
            "-vf",              f"scale={w}:{h}",   # resize inside ffmpeg (free)
            "-pix_fmt",         "bgr24",
            "-f",               "rawvideo",
            "-",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self._frame_bytes = w * h * 3

        t = threading.Thread(target=self._read, daemon=True)
        t.start()

    def _read(self):
        while not self._stopped:
            raw = self._proc.stdout.read(self._frame_bytes)
            if len(raw) != self._frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.h, self.w, 3)).copy()
            with self._lock:
                self._frame = frame  # always overwrites — main loop gets the freshest frame

    def read(self):
        with self._lock:
            return self._frame

    def release(self):
        self._stopped = True
        self._proc.terminate()


# ─────────────────────────────────────────────
# HOMOGRAPHY & UTILS
# ─────────────────────────────────────────────
def compute_homography(corner_px):
    src, dst = [], []
    for tid, px in corner_px.items():
        src.append(px)
        dst.append(CORNER_TAGS[tid])
    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(
        np.array(src, dtype=np.float32),
        np.array(dst, dtype=np.float32)
    )
    return H


def pixel_to_field(H, px, py):
    r = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), H)[0][0]
    col = int(round(max(0, min(GRID_COLS, r[0]))))
    row = int(round(max(0, min(GRID_ROWS, r[1]))))
    return col, row


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    detector = Detector(
        families="tag36h11",
        nthreads=2,          # old hardware: fewer threads = less context switching
        quad_decimate=2.0,
        quad_sigma=0.8,
        decode_sharpening=0.25,
        refine_edges=0,      # skip edge refinement — not needed for basic detection
    )

    stream = FFmpegStream(RTSP_URL, DETECT_W, DETECT_H)
    H = None
    prev_time = time.time()
    print(f"Started: {RTSP_URL}  (detect {DETECT_W}x{DETECT_H} → display {DISPLAY_W}x{DISPLAY_H})")

    try:
        while True:
            frame = stream.read()
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)

            # Update homography from corner tags (detect-res coords)
            corner_px = {
                d.tag_id: (d.center[0], d.center[1])
                for d in detections if d.tag_id in CORNER_TAGS
            }
            if len(corner_px) == 4:
                H = compute_homography(corner_px)

            # Upscale for display (INTER_NEAREST is fastest)
            display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)

            for det in detections:
                tid = det.tag_id
                ctr = (int(det.center[0] * SX), int(det.center[1] * SY))
                corners = (det.corners * [SX, SY]).astype(int).reshape(-1, 1, 2)

                if tid in CORNER_TAGS:  color, label = (180, 100, 255), f"C{tid}"
                elif tid in ROBOT_TAGS: color, label = (0, 180, 255),   ROBOT_TAGS[tid]
                elif tid in ITEM_TAGS:  color, label = (0, 210, 130),   ITEM_TAGS[tid]
                else:                   color, label = (200, 200, 200), f"ID{tid}"

                cv2.polylines(display, [corners], True, color, 2)

                if H is not None:
                    col, row = pixel_to_field(H, det.center[0], det.center[1])
                    label += f" [{col},{row}]"

                cv2.putText(display, label, (ctr[0] - 20, ctr[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            now = time.time()
            fps = 1.0 / (now - prev_time + 1e-9)
            prev_time = now
            cv2.putText(display, f"FPS:{fps:.0f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Robot Tracker", display)
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
