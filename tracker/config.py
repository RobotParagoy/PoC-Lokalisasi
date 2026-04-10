"""Configuration constants for the Robot AprilTag Tracking System."""

# ── Source Mode ──
RTSP_URL     = "rtsp://admin:admin@192.168.0.172:8554/Streaming/Channels/101"
TEST_VIDEO   = None
CAMERA_INDEX = 1

# ── Camera / Stream Settings ──
FRAME_W      = 1280
FRAME_H      = 720
TARGET_FPS   = 1
BRIGHTNESS   = 120
CONTRAST     = 1.3
PROCESS_FPS  = 4

# ── Display Settings ──
DISPLAY_W    = 960
DISPLAY_H    = 540

# ── Fisheye Undistortion ──
UNDISTORT_FISHEYE  = True
FISHEYE_K          = -0.5
FISHEYE_BALANCE    = 0.9
FISHEYE_K_STEP     = 0.05
FISHEYE_BAL_STEP   = 0.05

# ── RTSP Options ──
RTSP_RECONNECT_DELAY = 3
RTSP_TRANSPORT       = "tcp"
RTSP_SOCKET_TIMEOUT  = 10_000_000

# ── AprilTag Settings ──
TAG_FAMILY   = "tag36h11"

# ── Grid Dimensions ──
GRID_COLS = 8
GRID_ROWS = 4

# ── Field Quadrilateral (TL, TR, BR, BL) ──
FIELD_QUAD = [[370, 200], [2070, 240], [2040, 1110], [330, 1040]]
QUAD_STEP  = 10

# ── Tag Registrations ──
DOCKING_TAGS = {100: "docking1", 101: "docking2"}
ROBOT_TAGS   = {577: "Robot 1", 579: "Robot 2"}
ITEM_TAGS    = {580: "Item 1", 581: "Item 2", 582: "Item 3", 583: "Item 4",
                585: "Item 5", 586: "Item 6", 584: "Item 7", 576: "Item 8"}

CORNER_NAMES = ["TL", "TR", "BR", "BL"]
