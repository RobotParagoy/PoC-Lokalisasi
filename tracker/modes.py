"""Run modes — video file playback and live stream (RTSP / USB)."""

import time

import cv2

from tracker.config import (
    TEST_VIDEO, RTSP_URL, FRAME_W, FRAME_H,
    DISPLAY_W, DISPLAY_H, UNDISTORT_FISHEYE,
    PROCESS_FPS, RTSP_RECONNECT_DELAY,
)
from tracker.field import quad_homography
from tracker.fisheye import build_undistort_maps
from tracker.overlay import draw_quad_hud, log_positions
from tracker.processing import process_frame, make_state, handle_keypress
from tracker.capture import open_rtsp, open_video_source, show_waiting
from tracker.grid import log_grid
from tracker.mqtt import mqtt_connect, mqtt_publish_grid, mqtt_disconnect


def run_video_mode(detector):
    """Run the tracker against a local video file."""
    cap = cv2.VideoCapture(TEST_VIDEO)
    if not cap.isOpened():
        print(f"Error: could not open video '{TEST_VIDEO}'")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Playing: {TEST_VIDEO}  ({fps_video:.0f} fps)  —  press 'q' to quit")

    mqtt_connect()

    state = make_state()
    state['H'] = quad_homography(state['quad'])

    last_log_ms  = -9999
    LOG_INTERVAL = 500

    if UNDISTORT_FISHEYE:
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or FRAME_W
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_H
        state['frame_size'] = (src_w, src_h)
        state['undistort_maps'] = build_undistort_maps(src_w, src_h, state['fisheye_k'], state['fisheye_bal'])
        print(f"Fisheye undistortion ON  K={state['fisheye_k']:.2f}  balance={state['fisheye_bal']:.2f}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        vis, detections, matrix, coord_dict = process_frame(
            frame, detector, state['quad'], state['H'],
            state['selected'], state['undistort_maps'])

        if pos_ms - last_log_ms >= LOG_INTERVAL:
            log_positions(detections, state['H'], timestamp_ms=pos_ms)
            log_grid(matrix, coord_dict)
            mqtt_publish_grid(coord_dict)
            last_log_ms = pos_ms

        display = cv2.resize(vis, (DISPLAY_W, DISPLAY_H))
        draw_quad_hud(display, state['selected'],
                      state['fisheye_k'] if UNDISTORT_FISHEYE else None,
                      state['fisheye_bal'] if UNDISTORT_FISHEYE else None)
        cv2.namedWindow("Robot Tracker — Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Robot Tracker — Video", DISPLAY_W, DISPLAY_H)
        cv2.imshow("Robot Tracker — Video", display)

        key = cv2.waitKey(1) & 0xFF
        if handle_keypress(key, state):
            break

    cap.release()
    cv2.destroyAllWindows()
    mqtt_disconnect()


def run_stream_mode(detector):
    """Run the tracker against RTSP or USB camera."""
    cap, source_label = open_video_source()

    mqtt_connect()

    state = make_state()
    state['H'] = quad_homography(state['quad'])

    prev_time            = time.time()
    last_log_t           = prev_time
    LOG_INTERVAL         = 0.5
    detections           = []
    consecutive_failures = 0
    first_frame          = True
    maps_built           = False
    frame_interval       = 1.0 / PROCESS_FPS
    next_process_at      = 0.0

    while True:
        wait_budget = max(0.0, next_process_at - time.time())
        if wait_budget > 0:
            cap.wait_for_frame(timeout=wait_budget)
            key = cv2.waitKey(1) & 0xFF
            if handle_keypress(key, state):
                break
            if time.time() < next_process_at:
                continue

        ret, frame = cap.read()

        if not ret or frame is None:
            consecutive_failures += 1
            show_waiting(source_label)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if RTSP_URL is not None and consecutive_failures < 10:
                cap.wait_for_frame(timeout=0.1)
                continue

            if RTSP_URL is not None:
                print(f"\nRTSP stream lost — reconnecting in {RTSP_RECONNECT_DELAY}s ...")
                cap.release()
                time.sleep(RTSP_RECONNECT_DELAY)
                cap = open_rtsp(RTSP_URL)
                maps_built = False
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

        if UNDISTORT_FISHEYE and not maps_built:
            src_h, src_w = frame.shape[:2]
            state['frame_size'] = (src_w, src_h)
            state['undistort_maps'] = build_undistort_maps(src_w, src_h, state['fisheye_k'], state['fisheye_bal'])
            maps_built = True
            print(f"Fisheye undistortion ON  K={state['fisheye_k']:.2f}  balance={state['fisheye_bal']:.2f}")

        consecutive_failures = 0
        now = time.time()
        dt  = now - prev_time
        prev_time = now
        next_process_at = now + frame_interval

        vis, detections, matrix, coord_dict = process_frame(
            frame, detector, state['quad'], state['H'],
            state['selected'], state['undistort_maps'])

        if now - last_log_t >= LOG_INTERVAL:
            log_positions(detections, state['H'])
            log_grid(matrix, coord_dict)
            mqtt_publish_grid(coord_dict)
            last_log_t = now

        fps = 1.0 / dt if dt > 0 else 0

        display = cv2.resize(vis, (DISPLAY_W, DISPLAY_H))
        cv2.putText(display, f"{fps:.1f} fps", (DISPLAY_W - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        draw_quad_hud(display, state['selected'],
                      state['fisheye_k'] if UNDISTORT_FISHEYE else None,
                      state['fisheye_bal'] if UNDISTORT_FISHEYE else None)
        cv2.namedWindow(source_label, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(source_label, DISPLAY_W, DISPLAY_H)
        cv2.imshow(source_label, display)

        key = cv2.waitKey(1) & 0xFF
        if handle_keypress(key, state):
            break

    cap.release()
    cv2.destroyAllWindows()
    mqtt_disconnect()
    print(f"\nFinal quad config:")
    print(f"FIELD_QUAD = {state['quad']}")
    if UNDISTORT_FISHEYE:
        print(f"Final fisheye  K = {state['fisheye_k']:+.2f}   balance = {state['fisheye_bal']:.2f}")
