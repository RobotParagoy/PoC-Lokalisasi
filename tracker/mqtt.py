"""MQTT client — connect to broker and publish the grid matrix as JSON."""

import json
import threading
import time

import paho.mqtt.client as mqtt

from tracker.config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, ROBOT_TAGS

# ── Module-level singleton ───────────────────────────────────────────────────
_client = None           # type: mqtt.Client | None
_connected = False
_lock = threading.Lock()
_last_entity_change = {}  # (type, name) -> change key
_last_entity_info = {}    # (type, name) -> {id, name, type}
_last_entity_publish_time = {} # (type, name) -> ts_ms


def _on_connect(client, userdata, flags, rc, properties=None):
    global _connected
    if rc == 0:
        _connected = True
        print(f"[MQTT] Connected to {MQTT_BROKER}:{MQTT_PORT}")
    else:
        _connected = False
        print(f"[MQTT] Connection failed  rc={rc}")


def _on_disconnect(client, userdata, flags, rc, properties=None):
    global _connected
    _connected = False
    print(f"[MQTT] Disconnected  rc={rc}")


def mqtt_connect():
    """Create the MQTT client and start the background network loop.

    Safe to call more than once — subsequent calls are no-ops.
    """
    global _client
    with _lock:
        if _client is not None:
            return _client

        _client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="robot-tracker",
            protocol=mqtt.MQTTv311,
        )
        _client.on_connect    = _on_connect
        _client.on_disconnect = _on_disconnect

        try:
            _client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            _client.loop_start()          # non-blocking background thread
            print(f"[MQTT] Connecting to {MQTT_BROKER}:{MQTT_PORT} …")
        except Exception as exc:
            print(f"[MQTT] Could not connect: {exc}")
            _client = None

    return _client


def _publish_json(topic, payload):
    if _client is None or not _connected:
        return
    _client.publish(topic, json.dumps(payload), qos=0)


def _entity_change_key(entity):
    grid = entity.get("grid") or {}
    return (
        True,
        grid.get("col"),
        grid.get("row"),
        entity.get("orientation_deg"),
    )


def _name_to_topic(name, fallback_prefix, fallback_id):
    if name:
        return name.strip().lower().replace(" ", "_")
    return f"{fallback_prefix}_{fallback_id}"


def mqtt_publish_state(coord_dict, entities, ts_ms=None, base_topic=None):
    """Publish aggregate state plus per-entity updates on change."""
    if _client is None or not _connected:
        return

    if ts_ms is None:
        ts_ms = int(time.time() * 1000)

    base = base_topic or MQTT_TOPIC

    robots = {}
    items = {}
    for entity in entities:
        if entity.get("type") == "robot":
            robots[str(entity["id"])] = entity
        elif entity.get("type") == "item":
            items[str(entity["id"])] = entity

    state_payload = {
        "ts_ms": int(ts_ms),
        "grid": coord_dict,
        "robots": robots,
        "items": items,
    }
    _publish_json(f"{base}/state", state_payload)

    seen = set()
    for entity in entities:
        ent_type = entity.get("type")
        if ent_type not in ("robot", "item"):
            continue

        ent_id = int(entity["id"])
        ent_name = entity.get("name")
        topic_name = _name_to_topic(ent_name, ent_type, ent_id)
        key = (ent_type, topic_name)
        seen.add(key)

        change_key = _entity_change_key(entity)
        
        last_pub_time = _last_entity_publish_time.get(key, 0)
        should_publish = (_last_entity_change.get(key) != change_key)
        if ent_type == "robot" and (ts_ms - last_pub_time) >= 10000:
            should_publish = True
        
        if should_publish:
            payload = dict(entity)
            payload["visible"] = True
            payload["ts_ms"] = int(ts_ms)
            _publish_json(f"{base}/{ent_type}s/{topic_name}", payload)

            _last_entity_change[key] = change_key
            _last_entity_publish_time[key] = int(ts_ms)
            _last_entity_info[key] = {
                "id": ent_id,
                "name": ent_name,
                "type": ent_type,
            }

    robot_names = {
        _name_to_topic(name, "robot", tid): name
        for tid, name in ROBOT_TAGS.items()
    }
    for topic_name, robot_name in robot_names.items():
        key = ("robot", topic_name)
        if key in seen:
            continue

        change_key = (False, None, None, None)
        
        last_pub_time = _last_entity_publish_time.get(key, 0)
        should_publish = (_last_entity_change.get(key) != change_key) or ((ts_ms - last_pub_time) >= 10000)

        if should_publish:
            payload = {
                "id": None,
                "name": robot_name,
                "type": "robot",
                "visible": False,
                "ts_ms": int(ts_ms),
            }
            _publish_json(f"{base}/robots/{topic_name}", payload)
            _last_entity_change[key] = change_key
            _last_entity_publish_time[key] = int(ts_ms)
            _last_entity_info[key] = {
                "id": None,
                "name": robot_name,
                "type": "robot",
            }


def mqtt_publish_grid(coord_dict):
    """Publish the grid coordinate dict as JSON on the configured topic.

    Silently skips if the client is not connected.
    """
    if _client is None or not _connected:
        return

    payload = json.dumps(coord_dict)
    _client.publish(MQTT_TOPIC, payload, qos=0)


def mqtt_disconnect():
    """Gracefully stop the network loop and disconnect."""
    global _client, _connected
    with _lock:
        if _client is not None:
            _client.loop_stop()
            _client.disconnect()
            _client = None
            _connected = False
            print("[MQTT] Disconnected.")
