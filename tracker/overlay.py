"""Visualisation overlay — field quad, tag labels, HUD, and terminal log."""

import cv2
import numpy as np

from tracker.config import CORNER_NAMES, DISPLAY_H, GRID_CELL_SIZE_CM
from tracker.field import pixel_to_field, pixel_to_field_continuous
from tracker.tags import tag_orientation, classify_tag, tag_color


def draw_overlay(frame, detections, quad, H, selected, transformer=None):
    """Draw field quad, corner handles, and tag labels onto frame."""
    overlay = frame.copy()
    
    try:
        H_inv = np.linalg.inv(H)
    except:
        H_inv = None

    pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

    for i, (cx, cy) in enumerate(quad):
        is_sel = (i == selected)
        dot_color  = (0, 255, 255) if is_sel else (255, 255, 0)
        text_color = (0, 255, 255) if is_sel else (200, 200, 50)
        cv2.circle(overlay, (int(cx), int(cy)), 8 if is_sel else 5, dot_color, -1)
        cv2.putText(overlay, f"{i+1}:{CORNER_NAMES[i]}", (int(cx) + 10, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    for det in detections:
        tid  = det.tag_id
        ctr  = det.center.astype(int)
        corn = det.corners.astype(int)
        color = tag_color(tid)
        name, _ = classify_tag(tid)

        cv2.polylines(overlay, [corn.reshape(-1, 1, 2)], True, color, 2)

        # 3D Parallax visual pointer
        use_3d = False
        if transformer is not None and H_inv is not None:
            obj_height = transformer.get_height(name)
            if obj_height > 0.1:  # Only draw pointers for elevated objects
                world_coords = transformer.pixel_to_warehouse_world(ctr[0], ctr[1], obj_height)
                if world_coords:
                    gx = world_coords[0] / GRID_CELL_SIZE_CM
                    gy = world_coords[1] / GRID_CELL_SIZE_CM
                    floor_pt_grid = np.array([[[gx, gy]]], dtype=np.float32)
                    floor_px = cv2.perspectiveTransform(floor_pt_grid, H_inv)[0][0]
                    fx, fy = int(floor_px[0]), int(floor_px[1])
                    
                    # Draw a line from the actual tag in the air straight down to the floor projection
                    cv2.line(overlay, (ctr[0], ctr[1]), (fx, fy), (200, 50, 200), 2, cv2.LINE_AA)
                    cv2.circle(overlay, (fx, fy), 4, (0, 0, 255), -1)  # Red dot at the base
                    use_3d = True

        if use_3d:
            col, row = int(max(0, min(7, gx))), int(max(0, min(3, gy))) # Quick bounds constraint
        else:
            col, row = pixel_to_field(det.center[0], det.center[1], H)
            
        orient = tag_orientation(det.corners)
        label = f"{name} [{col},{row}] {orient}°"

        cv2.putText(overlay, label, (ctr[0] - 20, ctr[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    alpha = 0.85
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_quad_hud(display, selected, fisheye_k=None, fisheye_bal=None):
    """Draw corner-selection and move key reference HUD."""
    lines = [
        "Field quad adjust:",
        "1/2/3/4  select corner",
        "  1=TL  2=TR",
        "  4=BL  3=BR",
        "w/s  move up/down",
        "a/d  move left/right",
        f"Active: {selected+1} {CORNER_NAMES[selected]}",
    ]
    if fisheye_k is not None:
        lines.append("")
        lines.append("Fisheye correction:")
        lines.append(f"  +/-   K = {fisheye_k:+.2f}")
        lines.append(f"  [/]   balance = {fisheye_bal:.2f}")
    x = 10
    y = DISPLAY_H - 10 - len(lines) * 18
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i == len(lines) - 1 else (200, 200, 50)
        cv2.putText(display, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        y += 18


def log_positions(detections, H, timestamp_ms=None):
    """Print a position table for all detected tags."""
    print("\033[2J\033[H", end="")

    ts = f"  t = {timestamp_ms:.0f} ms" if timestamp_ms is not None else ""
    print(f"── Detected Tags {ts}")
    print(f"  {'ID':>4}  {'Name':<16}  {'Type':<8}  {'Pixel (x,y)':>18}  {'Grid (col,row)':>14}  {'Orient':>6}")
    print("  " + "─" * 80)

    for det in sorted(detections, key=lambda d: d.tag_id):
        tid = det.tag_id
        px, py = det.center
        name, tag_type = classify_tag(tid)
        col, row = pixel_to_field(px, py, H)
        grid_str = f"({col}, {row})"
        orient   = tag_orientation(det.corners)
        print(f"  {tid:>4}  {name:<16}  {tag_type:<8}  ({px:6.0f}, {py:6.0f})  {grid_str:>14}  {orient:>5}°")

    print("─" * 82)
