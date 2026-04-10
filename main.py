from tracker.config import TEST_VIDEO, RTSP_URL
from tracker.capture import create_detector
from tracker.modes import run_video_mode, run_stream_mode


def main():
    detector = create_detector()

    if TEST_VIDEO is not None and RTSP_URL is None:
        run_video_mode(detector)
    else:
        run_stream_mode(detector)


if __name__ == "__main__":
    main()
