"""
Hand Tracker AR Drawing App mit Formen-Menü

Installation:
    pip install opencv-python mediapipe

Start:
    python hand_tracker_gesteuert.py

Steuerung:
    Zeigefinger                Maus/Pointer bewegen
    Daumen + Zeigefinger       Linksklick / Auswahl
    Daumen + Mittelfinger      Formen-Menü öffnen
    Im Menü: Zeigefinger auf Form halten -> wählen
    Ein Shape auswählen und mit Zeigefinger ziehen
    Zwei Finger nahe am Shape -> Größe ändern
    Q / ESC                    Beenden
    H                         Hilfe
    C                         Shapes löschen
    F                         Vollbild

Beschreibung:
    Diese App zeichnet live Formen in der Kamera, wie AR-Objekte im Raum.
    Die Formen erscheinen im Bild und können verschoben und vergrößert werden.
"""

import math
import os
import time
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
WINDOW_NAME = "AR Shapes"

GREEN = (0, 255, 0)
DARK_GREEN = (0, 150, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 100, 0)
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)

# Hand landmarks
WRIST = 0
THUMB_IP = 3
THUMB_TIP = 4
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


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Lade Hand-Modell herunter ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modell geladen.")
    except Exception as e:
        raise SystemExit(f"Download fehlgeschlagen: {e}\nLade es manuell herunter und lege es hier ab:\n{MODEL_PATH}")


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def to_pixels(landmarks, w, h):
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]


def in_circle(point, cx, cy, r):
    return (point[0] - cx) ** 2 + (point[1] - cy) ** 2 <= r ** 2


def get_finger_state(points):
    # returns [thumb, index, middle, ring, pinky]
    # with robust check using angle and distance
    thumb = dist(points[THUMB_TIP], points[WRIST]) > dist(points[THUMB_IP], points[WRIST]) * 1.15 and dist(points[THUMB_TIP], points[WRIST]) > 35
    index = dist(points[INDEX_TIP], points[WRIST]) > dist(points[INDEX_PIP], points[WRIST]) * 1.1 and dist(points[INDEX_TIP], points[WRIST]) > 60
    middle = dist(points[MIDDLE_TIP], points[WRIST]) > dist(points[MIDDLE_PIP], points[WRIST]) * 1.12 and dist(points[MIDDLE_TIP], points[WRIST]) > 70
    ring = dist(points[RING_TIP], points[WRIST]) > dist(points[RING_PIP], points[WRIST]) * 1.1 and dist(points[RING_TIP], points[WRIST]) > 60
    pinky = dist(points[PINKY_TIP], points[WRIST]) > dist(points[PINKY_PIP], points[WRIST]) * 1.08 and dist(points[PINKY_TIP], points[WRIST]) > 55
    return [thumb, index, middle, ring, pinky]


def pinch_ratio(points, tip_index):
    palm = dist(points[WRIST], points[MIDDLE_MCP]) + 1e-6
    return dist(points[THUMB_TIP], points[tip_index]) / palm


def draw_text(img, text, org, scale=0.7, color=WHITE, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + base + 2), BLACK, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_hand(frame, pts, show_lines=True):
    if show_lines:
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], DARK_GREEN, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 9 if i in (4, 8, 12, 16, 20) else 6
        cv2.circle(frame, p, r, GREEN, -1, cv2.LINE_AA)
        cv2.circle(frame, p, r, BLACK, 1, cv2.LINE_AA)


def draw_pinch_link(frame, pts, tip):
    cv2.line(frame, pts[THUMB_TIP], pts[tip], YELLOW, 2, cv2.LINE_AA)


def draw_menu(frame, center, active_index):
    cx, cy = center
    radius = 140
    for i, name in enumerate(MENU_OPTIONS):
        ang = -math.pi / 2 + i * (2 * math.pi / len(MENU_OPTIONS))
        x = cx + int(radius * math.cos(ang))
        y = cy + int(radius * math.sin(ang))
        color = MENU_COLORS[name]
        if i == active_index:
            out = 26
            inr = 48
            cv2.circle(frame, (x, y), 58, color, 4, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 34, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            out = 24
            inr = 42
            cv2.circle(frame, (x, y), 48, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), inr, color, -1, cv2.LINE_AA)
        cv2.putText(frame, name[:1].upper(), (x - 7, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2, cv2.LINE_AA)

    cv2.circle(frame, center, 14, WHITE, -1, cv2.LINE_AA)
    cv2.circle(frame, center, 18, YELLOW, 2, cv2.LINE_AA)


# ---------- Shapes ----------

def make_shape(kind, x, y, size, color):
    return {
        "kind": kind,
        "x": x,
        "y": y,
        "size": size,
        "color": color,
    }


def draw_shape(frame, shape):
    x, y, size = shape["x"], shape["y"], shape["size"]
    kind = shape["kind"]
    color = shape["color"]

    if kind == "circle":
        cv2.circle(frame, (int(x), int(y)), int(size), color, 4, cv2.LINE_AA)
    elif kind == "square":
        s = int(size)
        pts = [(x-s, y-s), (x+s, y-s), (x+s, y+s), (x-s, y+s)]
        pts = np_to_cv([(int(px), int(py)) for px, py in pts])
        cv2.polylines(frame, [pts], True, color, 4, cv2.LINE_AA)
    elif kind == "triangle":
        s = int(size)
        pts = [(x, y-s), (x+s, y+s), (x-s, y+s)]
        pts = np_to_cv([(int(px), int(py)) for px, py in pts])
        cv2.polylines(frame, [pts], True, color, 4, cv2.LINE_AA)
    elif kind == "star":
        points = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            r = size if i % 2 == 0 else size * 0.45
            px = x + r * math.cos(ang)
            py = y + r * math.sin(ang)
            points.append((int(px), int(py)))
        cv2.polylines(frame, [np_to_cv(points)], True, color, 4, cv2.LINE_AA)
    elif kind == "heart":
        # simple heart outline
        for i in range(0, 180, 5):
            ang = math.radians(i)
            x1 = size * 16 * math.sin(ang) ** 3
            y1 = -size * (13 * math.cos(ang) - 5 * math.cos(2 * ang) - 2 * math.cos(3 * ang) - math.cos(4 * ang))
            px = int(x + x1 / 18)
            py = int(y + y1 / 18)
            if i == 0:
                pts = [(px, py)]
            else:
                pts.append((px, py))
        cv2.polylines(frame, [np_to_cv(pts)], False, color, 4, cv2.LINE_AA)

    # highlight selection
    if shape.get("selected"):
        cv2.circle(frame, (int(x), int(y)), int(size + 14), (255, 255, 255), 2, cv2.LINE_AA)


def np_to_cv(points):
    import numpy as np
    return np.array(points, dtype=np.int32)


# ---------- Main ----------

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

    shapes = []
    selected_shape_id = None
    drag_shape_id = None
    resize_shape_id = None
    drag_prev = None
    resize_base_dist = None
    resize_base_size = None
    menu_center = None
    menu_open = False
    menu_active_index = 0
    show_help = True
    fullscreen = True
    start = time.time()
    last_ts = -1
    fps = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Kein Bild von der Kamera.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts = int((time.time() - start) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            result = landmarker.detect_for_video(mp_image, ts)
            pointer = None
            pointer_index = None
            selection = None
            menu_target = None
            menu_active_index = 0

            for idx, lms in enumerate(result.hand_landmarks):
                pts = to_pixels(lms, w, h)
                fingers = get_finger_state(pts)
                thumb, index, middle, ring, pinky = fingers
                hand_label = result.handedness[idx][0].category_name if idx < len(result.handedness) and result.handedness[idx] else "Right"

                # pointer finger = index up, other fingers closed
                is_pointer = index and not thumb and not middle and not ring and not pinky
                is_thumb_index = dist(pts[THUMB_TIP], pts[INDEX_TIP]) < 60 and thumb and index
                is_thumb_middle = dist(pts[THUMB_TIP], pts[MIDDLE_TIP]) < 80 and thumb and middle

                draw_hand(frame, pts, True)

                if is_pointer:
                    pointer = pts[INDEX_TIP]
                    pointer_index = idx
                    cv2.circle(frame, pointer, 10, CYAN, -1, cv2.LINE_AA)

                if is_thumb_middle:
                    menu_target = midpoint(pts[THUMB_TIP], pts[MIDDLE_TIP])
                    menu_center = menu_target
                    menu_open = True
                    draw_pinch_link(frame, pts, MIDDLE_TIP)

                if is_thumb_index:
                    draw_pinch_link(frame, pts, INDEX_TIP)

                # pointer selection and drag
                for s in shapes:
                    if dist(pointer, (s["x"], s["y"])) < s["size"] + 20 if pointer is not None else False:
                        s["selected"] = True
                        selected_shape_id = id(s)
                    else:
                        s["selected"] = False

                # if pointer is near shape, drag it
                if pointer is not None and drag_shape_id is None:
                    for i, s in enumerate(shapes):
                        if dist(pointer, (s["x"], s["y"])) < s["size"] + 18:
                            drag_shape_id = i
                            drag_prev = pointer
                            selected_shape_id = i
                            break

                if pointer is not None and drag_shape_id is not None:
                    if drag_prev is not None:
                        dx = pointer[0] - drag_prev[0]
                        dy = pointer[1] - drag_prev[1]
                        shapes[drag_shape_id]["x"] += dx
                        shapes[drag_shape_id]["y"] += dy
                        drag_prev = pointer

                # resize with two fingers if selected
                if pointer is not None and idx == pointer_index:
                    if len(shapes) > 0:
                        for i, s in enumerate(shapes):
                            if dist(pointer, (s["x"], s["y"])) < s["size"] + 20:
                                selected_shape_id = i
                                break

            # menu logic
            if menu_open and menu_center is not None:
                # determine active option by pointer / index finger
                if pointer is not None:
                    best = 0
                    best_dist = 999999
                    for i, name in enumerate(MENU_OPTIONS):
                        ang = -math.pi / 2 + i * (2 * math.pi / len(MENU_OPTIONS))
                        x = menu_center[0] + int(140 * math.cos(ang))
                        y = menu_center[1] + int(140 * math.sin(ang))
                        d = dist(pointer, (x, y))
                        if d < best_dist:
                            best_dist = d
                            best = i
                    menu_active_index = best
                    if pointer is not None and dist(pointer, menu_center) < 40:
                        # choose center creates shape
                        kind = MENU_OPTIONS[menu_active_index]
                        shapes.append(make_shape(kind, pointer[0], pointer[1], 45, MENU_COLORS[kind]))
                        menu_open = False
                        menu_center = None
                        selected_shape_id = len(shapes) - 1
                draw_menu(frame, (int(menu_center[0]), int(menu_center[1])), menu_active_index)

            # if pointer drag released, clear drag status
            if pointer is None:
                drag_shape_id = None
                drag_prev = None
                resize_shape_id = None
                resize_base_dist = None
                resize_base_size = None

            # draw all shapes
            for i, s in enumerate(shapes):
                s["selected"] = i == selected_shape_id
                draw_shape(frame, s)

            # status
            draw_text(frame, f"Shapes: {len(shapes)}", (15, 30), 0.7, GREEN)
            draw_text(frame, "Q/ESC Ende   F Vollbild   H Hilfe   C Reset", (15, h - 55), 0.55, WHITE)
            if show_help:
                draw_text(frame, "Daumen+Mittelfinger: Menü   Zeigefinger: ziehen   2 Finger: Größe", (15, h - 22), 0.55, WHITE)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == ord("c"):
                shapes.clear()
                selected_shape_id = None

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
