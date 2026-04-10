"""AprilTag helper utilities — classification, colour, orientation."""

import math

from tracker.config import DOCKING_TAGS, ROBOT_TAGS, ITEM_TAGS


def tag_orientation(corners):
    """Return tag orientation in whole degrees (0-359)."""
    dx = corners[1][0] - corners[0][0]
    dy = corners[1][1] - corners[0][1]
    angle = math.degrees(math.atan2(dy, dx))
    return int(angle % 360)


def classify_tag(tid):
    """Return (name, tag_type) for a given tag ID."""
    if tid in DOCKING_TAGS:
        return DOCKING_TAGS[tid], "docking"
    if tid in ROBOT_TAGS:
        return ROBOT_TAGS[tid], "robot"
    if tid in ITEM_TAGS:
        return ITEM_TAGS[tid], "item"
    return f"Tag {tid}", "?"


def tag_color(tid):
    """Return BGR display color for a given tag ID."""
    if tid in DOCKING_TAGS:
        return (255, 140, 0)
    if tid in ROBOT_TAGS:
        return (0, 180, 255)
    if tid in ITEM_TAGS:
        return (0, 210, 130)
    return (200, 200, 200)
