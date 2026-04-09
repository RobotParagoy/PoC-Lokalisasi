import cv2
import numpy as np
from pupil_apriltags import Detector
import threading
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RTSP_URL   = "rtsp://admin:admin@192.168.0.172:8554/Streaming/Channels/102"

DISPLAY_W  = 960
DISPLAY_H  = 540
SCALE      = 0.4   # detection resolution scale

GRID_COLS  = 8
GRID_ROWS  = 4

CORNER_TAGS = {100:(0,0), 101:(0,GRID_ROWS), 102:(GRID_COLS,0), 103:(GRID_COLS,GRID_ROWS)}
ROBOT_TAGS  = {577:"Robot 1", 579:"Robot 2"}
ITEM_TAGS   = {580:"Item 1", 581:"Item 2", 582:"Item 3", 583:"Item 4", 585:"Item 5"}


# ─────────────────────────────────────────────
# PYTHON PORT OF Qengineering/RTSP-with-OpenCV
# ─────────────────────────────────────────────
class RTSPCapture:
    """
    Mirrors the C++ RTSPCapture pattern from Qengineering/RTSP-with-OpenCV.

    Key insight: cap.grab() advances the internal buffer WITHOUT decoding the
    frame (cheap). cap.retrieve() decodes only when you actually need the image
    (expensive). The background thread calls grab() in a tight loop so the
    buffer never accumulates stale frames. GetLatestFrame() calls retrieve()
    on demand — you always get the most recent image, never a queued one.
    """

    def __init__(self, url: str):
        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {url}")

        self._grabbed = False
        self._lock    = threading.Lock()
        self._stopped = False

        t = threading.Thread(target=self._grab_loop, daemon=True)
        t.start()

    def _grab_loop(self):
        """Continuously grab (not decode) frames to keep buffer current."""
        while not self._stopped:
            grabbed = self._cap.grab()
            with self._lock:
                self._grabbed = grabbed

    def GetLatestFrame(self):
        """
        Decode and return the most recently grabbed frame.
        Returns None if the stream is not ready yet.
        """
        with self._lock:
            if not self._grabbed:
                return None
            ret, frame = self._cap.retrieve()
        return frame if ret else None

    def release(self):
        self._stopped = True
        self._cap.release()


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
        nthreads=2,
        quad_decimate=2.0,
        quad_sigma=0.8,
        decode_sharpening=0.25,
        refine_edges=0,
    )

    cap = RTSPCapture(RTSP_URL)
    H = None
    prev_time = time.time()
    print(f"Started: {RTSP_URL}")

    try:
        while True:
            frame = cap.GetLatestFrame()
            if frame is None:
                continue

            # Downscale for detection
            small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)

            # Scale coords back to full-frame
            inv = 1.0 / SCALE
            corner_px = {}
            for det in detections:
                det.center  *= inv
                det.corners *= inv
                if det.tag_id in CORNER_TAGS:
                    corner_px[det.tag_id] = (det.center[0], det.center[1])

            if len(corner_px) == 4:
                H = compute_homography(corner_px)

            # Draw
            display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            sx = DISPLAY_W / frame.shape[1]
            sy = DISPLAY_H / frame.shape[0]

            for det in detections:
                tid = det.tag_id
                ctr = (int(det.center[0] * sx), int(det.center[1] * sy))
                corners = (det.corners * [sx, sy]).astype(int).reshape(-1, 1, 2)

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

            cv2.imshow("Robot Tracker [RTSP]", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                H = None
                print("Homography reset")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
