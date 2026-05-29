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

## Output Payload

The state of the system is serialized into a lightweight JSON payload and published over the network. The payload is represented as a Key-Value pair dictionary mapping grid coordinates to cell contents.

### Example Payload Structure
```json
{
  "(0,0)": 0,
  "(1,0)": 0,
  "(0,1)": 4,
  "(0,2)": 4,
  "(7,3)": [577, 90],
  "(4,2)": [583, 180]
}
```

### Payload Anatomy
- **Key:** A string representing the cell's coordinate `"(col,row)"`.
- **Value:** The content of the cell. Values can be either integers (representing static grid slots) or an array (representing detection details).

**Static Integer Values:**
- `0`: **Empty** / unoccupied lane without any tags.
- `4`: **Docking area** (Hardcoded to permanent cells `(0,1)` and `(0,2)`).
- `1`: **Robot** (Used internally for matrix building).
- `2`: **Goods/Items** (Used internally for matrix building).

**Detection Values (Array):** 
Whenever a tag enters the grid, its assigned cell key receives a two-element array:
`[tag_id, orientation_degrees]`
- `tag_id` (int): The exact AprilTag ID detected (e.g., Robot ID `577`, Item ID `583`).
- `orientation_degrees` (int): Rotational heading of the tag bounding box from 0 to 359 degrees.

The object is assigned to a grid point using the nearest-cell-centre logic (Voronoi region assignment), effectively eliminating tracking dead zones. Objects also utilize true 3D spatial transformation that factors in the vertical plane (Z-heights of known tags) to prevent radial distortion as they move out from the camera center.

## Encoding & Delivery

The JSON payload is published using the **MQTT** protocol via `paho-mqtt`.

- **Encoding:** JSON-formatted text (UTF-8)
- **QoS:** `0` (At most once delivery)
- **Protocol:** `MQTTv311`

**Default MQTT Settings (Configurable in `tracker/config.py`):**
- **Broker IP:** `192.168.0.142`
- **Broker Port:** `1883`
- **Topic:** `local`
- **Client ID:** `robot-tracker`

## Operations

Visual debugging bindings available on frame render:
- `1`, `2`, `3`, `4`: Select quad corner
- Arrows `Up/Down/Left/Right`: Adjust quad position
- `+ / -`: Tune fisheye K distortion profile
- `[ / ]`: Tune fisheye balance
- `Q`: Quit application