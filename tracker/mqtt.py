"""MQTT client — connect to broker and publish the grid matrix as JSON."""

import json
import threading

import paho.mqtt.client as mqtt

from tracker.config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC

# ── Module-level singleton ───────────────────────────────────────────────────
_client = None           # type: mqtt.Client | None
_connected = False
_lock = threading.Lock()


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
