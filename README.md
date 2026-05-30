# PoC-Lokalisasi

**PoC-Lokalisasi** is a Proof of Concept (PoC) AprilTag-based tracking and localization system. It processes RTSP camera feeds (or local video) to track robots and items in a smart warehouse setting, compensates for fisheye lens distortion, and applies a 3D coordinate transformation based on the physical height of objects. 

The software maps physical locations onto an 8×4 grid layout and broadcasts the matrix of tracked objects dynamically via MQTT.

## Overview

- **Source Input:** RTSP stream (default `rtsp://admin:admin@192.168.0.172:8554/Streaming/Channels/101`) or Camera/Video source.
- **Processing:** Detects Tag36h11 family AprilTags. Includes perspective (homography) transformation, camera calibration/fisheye undistortion, and dynamic configuration via `tuned_config.json`.
- **Outputs:** Real-time grid mapping serialization sent to an MQTT broker.

## Grid & Coordinates

The area is represented by a virtually mapped occupancy grid.
- **Dimensions:** 8 Columns × 4 Rows
- **Cell Size:** 30 × 30 cm
- **Origin:** Column 0, Row 0 starts at the top-left depending on homography perspective calibration.

**Coordinate Schema:**
      col →  0  1  2  3  4  5  6  7
    row 0
    row 1        ← permanent docking (col 0)
    row 2        ← permanent docking (col 0)
    row 3

## MQTT Output

The system publishes a full snapshot every cycle and sends per-entity updates only when an entity changes or disappears. The legacy base topic remains available for backward compatibility.

### Aggregate Snapshot Topic

**Topic:** `{MQTT_TOPIC}/state` (default: `local/state`)

Payload:
```json
{
  "ts_ms": 1717040000000,
  "grid": {
    "(0,0)": 0,
    "(0,1)": 4,
    "(7,3)": [577, 90]
  },
  "robots": {
    "577": {
      "id": 577,
      "name": "Robot 1",
      "type": "robot",
      "grid": {"col": 7, "row": 3},
      "orientation_deg": 90,
      "pixel": {"x": 1260, "y": 640},
      "mapping": "3d",
      "world_cm": {"x": 210.5, "y": 95.2}
    }
  },
  "items": {
    "583": {
      "id": 583,
      "name": "Item 4",
      "type": "item",
      "grid": {"col": 4, "row": 2},
      "orientation_deg": 180,
      "pixel": {"x": 820, "y": 510},
      "mapping": "2d"
    }
  }
}
```

### Per-Entity Topics (Published on Change)

- **Robots:** `{MQTT_TOPIC}/robots/<robot_id>` (example: `local/robots/577`)
- **Items:** `{MQTT_TOPIC}/items/<item_id>` (example: `local/items/583`)

Each per-entity payload includes `visible` and `ts_ms`. When an entity disappears, `visible` becomes `false` and the last known ID/name are retained.

### Legacy Topic (Backward Compatibility)

**Topic:** `{MQTT_TOPIC}` (default: `local`)

Payload: the original grid-only JSON mapping. This keeps existing consumers working while new consumers can subscribe to detailed state and per-entity topics.

### Grid Encoding (Inside `grid`)

- **Key:** `"(col,row)"`
- **Value:** `0` (empty), `4` (docking), or `[tag_id, orientation_degrees]` for detected tags.

The grid assignment uses nearest-cell-centre logic (Voronoi region assignment), removing dead zones. When available, a 3D transform is used to map tag centers based on known object heights.

## Encoding & Delivery

The JSON payload is published using the **MQTT** protocol via `paho-mqtt`.

- **Encoding:** JSON-formatted text (UTF-8)
- **QoS:** `0` (At most once delivery)
- **Protocol:** `MQTTv311`

**Default MQTT Settings (Configurable in `tracker/config.py`):**
- **Broker IP:** `192.168.0.142`
- **Broker Port:** `1883`
- **Base Topic:** `local`
- **Client ID:** `robot-tracker`

## Operations

Visual debugging bindings available on frame render:
- `1`, `2`, `3`, `4`: Select quad corner
- Arrows `Up/Down/Left/Right`: Adjust quad position
- `+ / -`: Tune fisheye K distortion profile
- `[ / ]`: Tune fisheye balance
- `Q`: Quit application