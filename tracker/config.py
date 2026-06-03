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

# Physical dimension of a single grid cell in cm for calibration
GRID_CELL_SIZE_CM = 30.0

# ── Detection physical Heights (Z) in cm ──
# (Items are assigned automatically below)
OBJECT_HEIGHTS = {
    "Robot 1": 15.0,
    "Robot 2": 15.0,
    "Robot 3": 15.0,
    "Robot 4": 15.0,
    "docking1": 0.0,
    "docking2": 0.0,
}

# ── Detection Zone ──
# Fraction (0–1) of each cell's width/height used for the visual hot-zone box.
# Detection itself uses nearest-cell-centre assignment (no dead zones).
DETECTION_ZONE_RATIO = 0.65

# Extra shrink applied to the drawn hot-zone at the image edge vs. centre.
# 0.0 = no correction; 0.4 = 40 % extra shrink at the far edge.
FISHEYE_EDGE_SHRINK  = 0.3

# ── Field Quadrilateral (TL, TR, BR, BL) ──
FIELD_QUAD = [[370, 200], [2070, 240], [2040, 1110], [330, 1040]]
QUAD_STEP  = 10

# ── Tag Registrations ──
DOCKING_TAGS = {100: "docking1", 101: "docking2"}
ROBOT_TAGS   = {577: "Robot 1", 578: "Robot 2", 579: "Robot 3", 576: "Robot 4"}
ITEM_TAGS    = {580: "Item 1", 581: "Item 2", 582: "Item 3", 583: "Item 4",
                585: "Item 5", 586: "Item 6", 584: "Item 7", 576: "Item 8",
                560: "Item 9", 561: "Item 10", 562: "Item 11", 563: "Item 12",
                564: "Item 13", 565: "Item 14", 566: "Item 15", 567: "Item 16"}



CORNER_NAMES = ["TL", "TR", "BR", "BL"]

# ── MQTT Settings ──
MQTT_BROKER  = "192.168.0.142"
MQTT_PORT    = 1883
MQTT_TOPIC   = "local"

# Automatically assign the same height (30.0 cm) to all registered items
for item_name in ITEM_TAGS.values():
    if item_name not in OBJECT_HEIGHTS:
        OBJECT_HEIGHTS[item_name] = 30.0
