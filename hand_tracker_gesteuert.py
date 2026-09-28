"""
Hand Tracker AR Drawing App - verbessert, krass, mit Radialmenü und Gestensteuerung

Installation:
    pip install opencv-python mediapipe numpy

Start:
    python hand_tracker_gesteuert.py

Gesten:
    Faust kurz halten      Radialmenü öffnen
    Zeigefinger            Pointer bewegen
    Daumen + Zeigefinger   Form auswählen / verschieben
    Pinch (Daumen+Zeigefinger nah beieinander)  Größe ändern
    2 Finger              Rotation
    3 Finger              Form löschen
    C                     Alle Formen löschen
    F                     Vollbild an/aus
    H                     Hilfe an/aus
    Q / ESC               Beenden

Ziel:
    Zeichne Formen direkt in der Kamera, nutze ein modernes Radialmenü und interaktive Gesten.
"""

import math
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
WINDOW_NAME = "Hand Tracker AR Studio"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 140, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 100, 0)
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PINK = (255, 0, 180)

WRIST = 0
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_PIP = 14
RING_TIP = 16
PINKY_PIP = 18
PINKY_TIP = 20

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

MENU_OPTIONS = ["circle", "square", "triangle", "star", "heart"]
MENU_COLORS = {
    "circle": (0, 255, 255),
    "square": (255, 0, 255),
    "triangle": (0, 165, 255),
    "star": (255, 165, 0),
    "heart": (255, 0, 0),
}


def ensure_model() -> None:
    if os.path.exists(MODEL_PATH):
        return
    print("Lade Hand-Modell herunter ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modell geladen.")
    except Exception as exc:
        raise SystemExit(
            f"Download fehlgeschlagen: {exc}\n"
            f"Lade die Datei manuell herunter:\n{MODEL_URL}\n"
            f"und lege sie hier ab:\n{MODEL_PATH}"
        )


def dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def to_pixels(landmarks, w: int, h: int):
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[(i + 1) % n]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
        if intersect:
            inside = not inside
    return inside


def point_in_shape(pt, shape):
    x, y = pt
    cx, cy = shape["x"], shape["y"]
    size = shape["size"]
    kind = shape["kind"]

    if kind == "circle":
        return (x - cx) ** 2 + (y - cy) ** 2 <= (size * 1.3) ** 2
    if kind == "square":
        half = size * 1.1
        return abs(x - cx) <= half and abs(y - cy) <= half
    if kind == "triangle":
        return (x - cx) ** 2 + (y - cy) ** 2 <= (size * 1.7) ** 2
    if kind == "star":
        return (x - cx) ** 2 + (y - cy) ** 2 <= (size * 1.6) ** 2
    if kind == "heart":
        return (x - cx) ** 2 + (y - cy) ** 2 <= (size * 1.5) ** 2
    return False


@dataclass
class Shape:
    kind: str
    x: float
    y: float
    size: float
    color: Tuple[int, int, int]
    rotation: float = 0.0
    selected: bool = False

    def draw(self, frame):
        x, y, size = int(self.x), int(self.y), int(self.size)
        color = self.color
        kind = self.kind

        if kind == "circle":
            cv2.circle(frame, (x, y), size, color, 4, cv2.LINE_AA)
        elif kind == "square":
            verts = np.array([
                (-size, -size), (size, -size), (size, size), (-size, size)
            ], dtype=np.float32)
            verts = self._rotate_points(verts)
            pts = np.array([(x + vx, y + vy) for vx, vy in verts], dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 4, cv2.LINE_AA)
        elif kind == "triangle":
            verts = np.array([
                (0, -size), (size * 0.9, size * 1.1), (-size * 0.9, size * 1.1)
            ], dtype=np.float32)
            verts = self._rotate_points(verts)
            pts = np.array([(x + vx, y + vy) for vx, vy in verts], dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 4, cv2.LINE_AA)
        elif kind == "star":
            star_pts = []
            for i in range(10):
                ang = self.rotation + i * math.pi / 5
                r = size if i % 2 == 0 else size * 0.5
                px = x + r * math.cos(ang - math.pi / 2)
                py = y + r * math.sin(ang - math.pi / 2)
                star_pts.append((int(px), int(py)))
            cv2.polylines(frame, [np.array(star_pts, dtype=np.int32)], True, color, 4, cv2.LINE_AA)
        elif kind == "heart":
            points = []
            for i in range(0, 360, 5):
                ang = math.radians(i)
                sx = 16 * math.sin(ang) ** 3
                sy = -(13 * math.cos(ang) - 5 * math.cos(2 * ang) - 2 * math.cos(3 * ang) - math.cos(4 * ang))
                px = x + sx * size / 18
                py = y + sy * size / 18
                points.append((int(px), int(py)))
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, color, 4, cv2.LINE_AA)

        if self.selected:
            cv2.circle(frame, (x, y), size + 18, WHITE, 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), size + 24, color, 1, cv2.LINE_AA)

    def _rotate_points(self, verts):
        theta = self.rotation
        rot = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)]
        ])
        out = []
        for vx, vy in verts:
            xr = vx * rot[0][0] + vy * rot[0][1]
            yr = vx * rot[1][0] + vy * rot[1][1]
            out.append((xr, yr))
        return np.array(out, dtype=np.float32)


def draw_text(img, text, org, scale=0.7, color=WHITE, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 8), (x + tw + 6, y + base + 4), BLACK, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_hand(frame, pts, show_lines=True):
    if show_lines:
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], DARK_GREEN, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        radius = 9 if i in (4, 8, 12, 16, 20) else 6
        cv2.circle(frame, p, radius, GREEN, -1, cv2.LINE_AA)
        cv2.circle(frame, p, radius, BLACK, 1, cv2.LINE_AA)


def draw_menu(frame, center, active_index):
    if center is None:
        return
    cx, cy = center
    radius = 150
    halo = 180

    for i in range(5):
        ang = -math.pi / 2 + i * (2 * math.pi / len(MENU_OPTIONS))
        x = int(cx + radius * math.cos(ang))
        y = int(cy + radius * math.sin(ang))
        color = MENU_COLORS[MENU_OPTIONS[i]]
        if i == active_index:
            cv2.circle(frame, (cx, cy), halo, color, 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 54, WHITE, 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 42, color, 3, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 26, color, -1, cv2.LINE_AA)
        cv2.putText(frame, MENU_OPTIONS[i][0].upper(), (x - 7, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2, cv2.LINE_AA)

    cv2.circle(frame, center, 20, WHITE, -1, cv2.LINE_AA)
    cv2.circle(frame, center, 28, YELLOW, 2, cv2.LINE_AA)


def get_fingers(points):
    thumb = dist(points[THUMB_TIP], points[WRIST]) > dist(points[THUMB_IP], points[WRIST]) * 1.1 and dist(points[THUMB_TIP], points[WRIST]) > 35
    index = dist(points[INDEX_TIP], points[WRIST]) > dist(points[INDEX_PIP], points[WRIST]) * 1.1 and dist(points[INDEX_TIP], points[WRIST]) > 60
    middle = dist(points[MIDDLE_TIP], points[WRIST]) > dist(points[MIDDLE_PIP], points[WRIST]) * 1.12 and dist(points[MIDDLE_TIP], points[WRIST]) > 70
    ring = dist(points[RING_TIP], points[WRIST]) > dist(points[RING_PIP], points[WRIST]) * 1.1 and dist(points[RING_TIP], points[WRIST]) > 60
    pinky = dist(points[PINKY_TIP], points[WRIST]) > dist(points[PINKY_PIP], points[WRIST]) * 1.08 and dist(points[PINKY_TIP], points[WRIST]) > 55
    return [thumb, index, middle, ring, pinky]


def is_fist(fingers):
    return not any(fingers)


def nearest_shape_point(pointer, shapes):
    best_idx = None
    best_dist = 999999
    for idx, shape in enumerate(shapes):
        d = dist(pointer, (int(shape.x), int(shape.y)))
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx


def angle_between(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def main():
    ensure_model()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Kamera konnte nicht geöffnet werden.")
        return

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    shapes: List[Shape] = []
    selected_shape_index: Optional[int] = None
    drag_shape_index: Optional[int] = None
    drag_prev: Optional[Tuple[int, int]] = None
    pointer = None
    pointer_prev = None
    menu_open = False
    menu_center = None
    menu_active_index = 0
    show_help = True
    fullscreen = True
    start_time = time.time()
    last_ts = -1
    fps = 0.0
    prev_time = time.time()
    status_text = "Warte auf Hand..."
    menu_hold_timer = 0.0
    action_state = "idle"
    pinch_prev_dist = None
    rotation_prev_angle = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts = int((time.time() - start_time) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            result = landmarker.detect_for_video(mp_image, ts)
            pointer = None
            user_fist = False
            user_thumb_index = False
            user_pinch = False
            user_two_fingers = False
            user_three_fingers = False
            primary_hand = None
            primary_points = None
            status_text = "Warte auf Hand..."

            for idx, lms in enumerate(result.hand_landmarks):
                pts = to_pixels(lms, w, h)
                fingers = get_fingers(pts)
                thumb, index, middle, ring, pinky = fingers
                user_fist = is_fist(fingers)
                user_thumb_index = thumb and index and not middle and not ring and not pinky
                user_two_fingers = index and middle and not thumb and not ring and not pinky
                user_three_fingers = index and middle and ring and not thumb and not pinky

                if index and not thumb and not middle and not ring and not pinky:
                    pointer = pts[INDEX_TIP]
                    status_text = "Zeigefinger aktiv: Pointer"
                    cv2.circle(frame, pointer, 10, CYAN, -1, cv2.LINE_AA)

                if user_thumb_index:
                    status_text = "Daumen + Zeigefinger: Auswahl"
                    cv2.line(frame, pts[THUMB_TIP], pts[INDEX_TIP], YELLOW, 2, cv2.LINE_AA)

                if user_fist:
                    status_text = "Faust erkannt: Menü öffnen"
                    if primary_hand is None:
                        primary_hand = pts
                        primary_points = pts

                if user_two_fingers:
                    status_text = "2 Finger: Rotation"

                if user_three_fingers:
                    status_text = "3 Finger: Löschen"

                draw_hand(frame, pts, True)

                if primary_hand is None:
                    primary_hand = pts
                    primary_points = pts

            if primary_hand is not None and pointer is None:
                pointer = primary_hand[INDEX_TIP] if len(primary_hand) > INDEX_TIP else None

            if user_fist:
                menu_hold_timer += 0.016
                if menu_hold_timer > 0.35:
                    menu_open = True
                    menu_center = (w // 2, h // 2)
                    action_state = "menu"
            else:
                if menu_open and menu_center is not None and not user_fist:
                    menu_open = False
                    menu_center = None
                    action_state = "idle"
                menu_hold_timer = 0.0

            if menu_open and menu_center is not None and pointer is not None:
                best_i = 0
                best_d = 999999
                for i, kind in enumerate(MENU_OPTIONS):
                    ang = -math.pi / 2 + i * (2 * math.pi / len(MENU_OPTIONS))
                    x = menu_center[0] + int(140 * math.cos(ang))
                    y = menu_center[1] + int(140 * math.sin(ang))
                    d = dist(pointer, (x, y))
                    if d < best_d:
                        best_d = d
                        best_i = i
                menu_active_index = best_i
                status_text = f"Menü: {MENU_OPTIONS[menu_active_index]} | Mitte: Erstellen"

                if dist(pointer, menu_center) < 45:
                    kind = MENU_OPTIONS[menu_active_index]
                    shapes.append(Shape(kind=kind, x=w // 2, y=h // 2, size=45, color=MENU_COLORS[kind]))
                    selected_shape_index = len(shapes) - 1
                    menu_open = False
                    menu_center = None
                    action_state = "created"
                    status_text = f"Form '{kind}' erstellt!"

            if not menu_open and pointer is not None and primary_points is not None:
                if selected_shape_index is None:
                    selected_shape_index = nearest_shape_point(pointer, shapes)
                else:
                    if selected_shape_index is not None and selected_shape_index < len(shapes) and point_in_shape(pointer, {
                        "x": shapes[selected_shape_index].x,
                        "y": shapes[selected_shape_index].y,
                        "size": shapes[selected_shape_index].size,
                        "kind": shapes[selected_shape_index].kind,
                    }):
                        shapes[selected_shape_index].selected = True
                    else:
                        for idx, shape in enumerate(shapes):
                            shape.selected = False

                if user_thumb_index:
                    if drag_shape_index is None:
                        for idx, shape in enumerate(shapes):
                            if point_in_shape(pointer, {"x": shape.x, "y": shape.y, "size": shape.size, "kind": shape.kind}):
                                drag_shape_index = idx
                                drag_prev = pointer
                                selected_shape_index = idx
                                shapes[idx].selected = True
                                break
                    if drag_shape_index is not None and drag_prev is not None:
                        dx = pointer[0] - drag_prev[0]
                        dy = pointer[1] - drag_prev[1]
                        shapes[drag_shape_index].x += dx
                        shapes[drag_shape_index].y += dy
                        drag_prev = pointer

                if user_two_fingers and selected_shape_index is not None and selected_shape_index < len(shapes):
                    shape = shapes[selected_shape_index]
                    current_angle = angle_between(primary_points[INDEX_TIP], primary_points[MIDDLE_TIP])
                    if rotation_prev_angle is None:
                        rotation_prev_angle = current_angle
                    delta = current_angle - rotation_prev_angle
                    shape.rotation += math.radians(delta) * 0.7
                    rotation_prev_angle = current_angle

                if user_three_fingers and selected_shape_index is not None and selected_shape_index < len(shapes):
                    del shapes[selected_shape_index]
                    selected_shape_index = None
                    drag_shape_index = None
                    drag_prev = None
                    rotation_prev_angle = None
                    status_text = "Form gelöscht"

                if pointer_prev is not None and user_thumb_index and selected_shape_index is not None and selected_shape_index < len(shapes):
                    shape = shapes[selected_shape_index]
                    pinch_dist = dist(primary_points[THUMB_TIP], primary_points[INDEX_TIP])
                    if pinch_prev_dist is None:
                        pinch_prev_dist = pinch_dist
                    size_delta = (pinch_dist - pinch_prev_dist) * 0.65
                    shape.size = clamp(shape.size + size_delta, 20, 170)
                    pinch_prev_dist = pinch_dist

            if pointer is None:
                drag_shape_index = None
                drag_prev = None
                pinch_prev_dist = None
                rotation_prev_angle = None
            pointer_prev = pointer

            if menu_open and menu_center is not None:
                draw_menu(frame, menu_center, menu_active_index)

            for idx, shape in enumerate(shapes):
                if selected_shape_index == idx:
                    shape.selected = True
                else:
                    shape.selected = False
                shape.draw(frame)

            draw_text(frame, f"Status: {status_text}", (15, 25), 0.7, GREEN)
            draw_text(frame, f"Shapes: {len(shapes)}", (15, 55), 0.7, WHITE)
            draw_text(frame, f"FPS: {int(fps)}", (15, 85), 0.7, GREEN)
            draw_text(frame, "Vollbild: AN" if fullscreen else "Vollbild: AUS", (15, 115), 0.6, BLUE)

            if show_help:
                draw_text(frame, "Q/ESC Ende   F Vollbild   H Hilfe   C Reset", (15, h - 55), 0.55, WHITE)
                draw_text(frame, "Faust = Menü   Zeigefinger = Pointer   Daumen+Zeigefinger = Verschieben", (15, h - 22), 0.55, WHITE)

            now = time.time()
            inst = 1.0 / max(now - prev_time, 1e-6)
            fps = fps * 0.9 + inst * 0.1 if fps else inst
            prev_time = now

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("c"):
                shapes.clear()
                selected_shape_index = None
                drag_shape_index = None
                menu_open = False
                menu_center = None
                status_text = "Alle Formen gelöscht"

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
