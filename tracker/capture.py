"""Video capture — threaded reader, RTSP connection, source initialisation."""

import os
import time
import threading

import cv2
import numpy as np
from pupil_apriltags import Detector

from tracker.config import (
    RTSP_TRANSPORT, RTSP_SOCKET_TIMEOUT, RTSP_URL,
    FRAME_W, FRAME_H, TARGET_FPS, BRIGHTNESS,
    DISPLAY_W, DISPLAY_H, CAMERA_INDEX, TAG_FAMILY,
)


class ThreadedVideoCapture:
    """Continuously grab frames in a background thread, keeping only the latest."""

    def __init__(self, cap):
        self._cap       = cap
        self._lock      = threading.Lock()
        self._frame     = None
        self._ret       = False
        self._seq       = 0
        self._consumed  = 0
        self._new_frame = threading.Event()
        self._running   = True
        self._thread    = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret   = ret
                self._frame = frame
                self._seq  += 1
            if ret:
                self._new_frame.set()
            else:
                time.sleep(0.01)

    def read(self):
        """Return the most recent frame."""
        with self._lock:
            self._consumed = self._seq
            return self._ret, self._frame

    def has_new_frame(self):
        """True if the reader grabbed a frame we haven't consumed yet."""
        with self._lock:
            return self._seq > self._consumed

    def wait_for_frame(self, timeout=None):
        """Block until a new frame is available or timeout."""
        ready = self._new_frame.wait(timeout=timeout)
        self._new_frame.clear()
        return ready

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
    """Open an RTSP stream with TCP transport and threaded reader."""
    opts = []
    if RTSP_TRANSPORT == "tcp":
        opts.append("rtsp_transport;tcp")
    opts.append(f"stimeout;{RTSP_SOCKET_TIMEOUT}")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("  ⚠  VideoCapture failed to open — returning raw handle.")
        return ThreadedVideoCapture(cap)

    print("  Waiting for first RTSP frame (warm-up) ...")
    warmup_deadline = time.time() + RTSP_SOCKET_TIMEOUT / 1_000_000
    while time.time() < warmup_deadline:
        ret, frame = cap.read()
        if ret and frame is not None:
            print("  First frame received — starting threaded reader.")
            wrapper = ThreadedVideoCapture(cap)
            with wrapper._lock:
                wrapper._ret   = True
                wrapper._frame = frame
            return wrapper
        time.sleep(0.1)

    print("  ⚠  Warm-up timed out — starting threaded reader anyway.")
    return ThreadedVideoCapture(cap)


def show_waiting(window_name):
    """Display a 'Waiting for stream...' placeholder."""
    placeholder = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for stream...", (20, DISPLAY_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H)
    cv2.imshow(window_name, placeholder)


def open_video_source():
    """Open the appropriate video source and return (cap, label)."""
    if RTSP_URL is not None:
        print(f"Connecting to RTSP stream: {RTSP_URL}")
        cap = open_rtsp(RTSP_URL)
        label = "Robot Tracker — RTSP"
        print("RTSP stream opened (threaded reader) — press 'q' to quit.")
        return cap, label

    raw_cap = cv2.VideoCapture(CAMERA_INDEX)
    raw_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    raw_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    raw_cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    raw_cap.set(cv2.CAP_PROP_BRIGHTNESS,   BRIGHTNESS)
    cap = ThreadedVideoCapture(raw_cap)
    label = "Robot Tracker"
    print("Starting USB camera (threaded reader) — press 'q' to quit.")
    return cap, label


def create_detector():
    """Create and return the AprilTag detector."""
    return Detector(
        families=TAG_FAMILY,
        nthreads=2,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )
