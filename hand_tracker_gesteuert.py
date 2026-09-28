"""
Hand Tracker AR Drawing App - überarbeitete stabile Version

Installation:
    pip install opencv-python mediapipe numpy

Start:
    python hand_tracker_gesteuert.py

Steuerung:
    Faust schnell machen      Menü-Rad öffnen
    Zeigefinger              Pointer bewegen
    Zeigefinger auf Form     Form im Menü auswählen
    Daumen + Zeigefinger     Auswahl / Klick
    Zeigefinger auf Form     Form verschieben
    2 Finger auf Form        Größe ändern
    Q / ESC                  Beenden
    C                       Formen löschen
    F                       Vollbild an/aus
    H                       Hilfe an/aus

Ziel:
    Formen wie Kreis, Quadrat, Dreieck, Stern, Herz werden live in der Kamera dargestellt.
    Gleichzeitig wird angezeigt, was die Hand gerade erkennt.
"""

import math
import os
import time
import urllib.request
from datetime import datetime

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
WINDOW_NAME = "AR Drawing App"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 150, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 100, 0)
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)

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
        raise SystemExit(f"Download fehlgeschlagen: {e}\nLade die Datei manuell herunter:\n{MODEL_URL}\nund lege sie hier ab:\n{MODEL_PATH}")


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def to_pixels(landmarks, w, h):
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]


def point_in_shape(pt, shape):
    x, y = pt
    sx, sy = shape["x"], shape["y"]
    s = shape["size"]
    kind = shape["kind"]
    if kind == "circle":
        return (x - sx) ** 2 + (y - sy) ** 2 <= s ** 2
    if kind == "square":
        return abs(x - sx) <= s and abs(y - sy) <= s
    if kind == "triangle":
        return abs(x - sx) + abs(y - sy) <= s * 1.2
    if kind == "star":
        return (x - sx) ** 2 + (y - sy) ** 2 <= (s * 1.3) ** 2
    if kind == "heart":
        return (x - sx) ** 2 + (y - sy) ** 2 <= (s * 1.5) ** 2
    return False


def make_shape(kind, x, y, size, color):
    return {"kind": kind, "x": x, "y": y, "size": size, "color": color, "selected": False}


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


def draw_shape(frame, shape):
    x, y, size = int(shape["x"]), int(shape["y"]), int(shape["size"])
    kind = shape["kind"]
    color = shape["color"]

    if kind == "circle":
        cv2.circle(frame, (x, y), size, color, 4, cv2.LINE_AA)
    elif kind == "square":
        cv2.rectangle(frame, (x - size, y - size), (x + size, y + size), color, 4, cv2.LINE_AA)
    elif kind == "triangle":
        pts = np.array([(x, y - size), (x + size, y + size), (x - size, y + size)], dtype=np.int32)
        cv2.polylines(frame, [pts], True, color, 4, cv2.LINE_AA)
    elif kind == "star":
        star_pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            r = size if i % 2 == 0 else size * 0.45
            px = x + r * math.cos(ang)
            py = y + r * math.sin(ang)
            star_pts.append((int(px), int(py)))
        cv2.polylines(frame, [np.array(star_pts, dtype=np.int32)], True, color, 4, cv2.LINE_AA)
    elif kind == "heart":
        pts = []
        for i in range(0, 360, 5):
            ang = math.radians(i)
            sx = 16 * math.sin(ang) ** 3
            sy = -(13 * math.cos(ang) - 5 * math.cos(2 * ang) - 2 * math.cos(3 * ang) - math.cos(4 * ang))
            px = int(x + sx * size / 18)
            py = int(y + sy * size / 18)
            pts.append((px, py))
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, color, 4, cv2.LINE_AA)

    if shape.get("selected"):
        cv2.circle(frame, (x, y), size + 12, (255, 255, 255), 2, cv2.LINE_AA)


def draw_menu(frame, center, active_index):
    if center is None:
        return
    cx, cy = center
    radius = 150
    for i, kind in enumerate(MENU_OPTIONS):
        ang = -math.pi / 2 + i * (2 * math.pi / len(MENU_OPTIONS))
        x = int(cx + radius * math.cos(ang))
        y = int(cy + radius * math.sin(ang))
        color = MENU_COLORS[kind]
        cv2.circle(frame, (x, y), 42, color, 3, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 24, color, -1, cv2.LINE_AA)
        cv2.putText(frame, kind[0].upper(), (x - 7, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2, cv2.LINE_AA)
        if i == active_index:
            cv2.circle(frame, (x, y), 52, WHITE, 2, cv2.LINE_AA)
    cv2.circle(frame, center, 20, WHITE, -1, cv2.LINE_AA)
    cv2.circle(frame, center, 28, YELLOW, 2, cv2.LINE_AA)


def get_fingers(points):
    thumb = dist(points[THUMB_TIP], points[WRIST]) > dist(points[THUMB_IP], points[WRIST]) * 1.15 and dist(points[THUMB_TIP], points[WRIST]) > 35
    index = dist(points[INDEX_TIP], points[WRIST]) > dist(points[INDEX_PIP], points[WRIST]) * 1.1 and dist(points[INDEX_TIP], points[WRIST]) > 60
    middle = dist(points[MIDDLE_TIP], points[WRIST]) > dist(points[MIDDLE_PIP], points[WRIST]) * 1.12 and dist(points[MIDDLE_TIP], points[WRIST]) > 70
    ring = dist(points[RING_TIP], points[WRIST]) > dist(points[RING_PIP], points[WRIST]) * 1.1 and dist(points[RING_TIP], points[WRIST]) > 60
    pinky = dist(points[PINKY_TIP], points[WRIST]) > dist(points[PINKY_PIP], points[WRIST]) * 1.08 and dist(points[PINKY_TIP], points[WRIST]) > 55
    return [thumb, index, middle, ring, pinky]


def is_fist(fingers):
    return not any(fingers)


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
    selected_shape_index = None
    drag_shape_index = None
    drag_prev = None
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
            user_thumb_middle = False
            status_text = "Warte auf Hand..."

            for idx, lms in enumerate(result.hand_landmarks):
                pts = to_pixels(lms, w, h)
                fingers = get_fingers(pts)
                thumb, index, middle, ring, pinky = fingers
                user_fist = is_fist(fingers)

                draw_hand(frame, pts, True)

                # Pointer: Zeigefinger allein
                if index and not thumb and not middle and not ring and not pinky:
                    pointer = pts[INDEX_TIP]
                    status_text = "Zeigefinger aktiv: Pointer"
                    cv2.circle(frame, pointer, 10, CYAN, -1, cv2.LINE_AA)

                # Auswahl: Daumen + Zeigefinger
                if thumb and index and not middle and not ring and not pinky:
                    user_thumb_index = True
                    status_text = "Daumen + Zeigefinger: Auswahl"
                    cv2.line(frame, pts[THUMB_TIP], pts[INDEX_TIP], YELLOW, 2, cv2.LINE_AA)

                # Menü: Faust
                if user_fist:
                    status_text = "Faust erkannt: Menü öffnen"
                    menu_open = True
                    menu_center = (w // 2, h // 2)

                # Drag / Resize logic wenn nicht im Menü
                if pointer is not None and not menu_open:
                    for i, shape in enumerate(shapes):
                        if point_in_shape(pointer, shape):
                            shape["selected"] = True
                            selected_shape_index = i
                        else:
                            shape["selected"] = False

                    if drag_shape_index is None:
                        for i, shape in enumerate(shapes):
                            if point_in_shape(pointer, shape):
                                drag_shape_index = i
                                drag_prev = pointer
                                break

                    if drag_shape_index is not None and drag_prev is not None:
                        dx = pointer[0] - drag_prev[0]
                        dy = pointer[1] - drag_prev[1]
                        shapes[drag_shape_index]["x"] += dx
                        shapes[drag_shape_index]["y"] += dy
                        drag_prev = pointer

                if user_thumb_index and pointer is not None and not menu_open:
                    best_idx = None
                    best_d = 999999
                    for i, shape in enumerate(shapes):
                        d = dist(pointer, (shape["x"], shape["y"]))
                        if d < best_d:
                            best_d = d
                            best_idx = i
                    if best_idx is not None:
                        selected_shape_index = best_idx
                        shapes[best_idx]["selected"] = True

                # Zwei Finger auf Form = Größenänderung
                if pointer is not None and len(shapes) > 0 and not menu_open:
                    for i, shape in enumerate(shapes):
                        if point_in_shape(pointer, shape):
                            d = dist(pointer, (shape["x"], shape["y"]))
                            if d < 120:
                                shape["size"] = max(25, min(110, int(d * 0.7)))

            if pointer is None:
                drag_shape_index = None
                drag_prev = None

            # Menü Logik
            if menu_open and menu_center is not None:
                if pointer is not None:
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
                    status_text = f"Menü: {MENU_OPTIONS[menu_active_index]} ausgewählt"

                    # Zeigefinger in die Mitte = Form erstellen
                    if dist(pointer, menu_center) < 35:
                        kind = MENU_OPTIONS[menu_active_index]
                        shapes.append(make_shape(kind, w // 2, h // 2, 40, MENU_COLORS[kind]))
                        menu_open = False
                        menu_center = None
                        selected_shape_index = len(shapes) - 1
                        status_text = f"Form '{kind}' erstellt!"

                draw_menu(frame, menu_center, menu_active_index)
            else:
                # Menü schließen wenn keine Faust mehr
                if not user_fist:
                    menu_open = False
                    menu_center = None

            for i, shape in enumerate(shapes):
                shape["selected"] = i == selected_shape_index
                draw_shape(frame, shape)

            draw_text(frame, f"Status: {status_text}", (15, 25), 0.7, GREEN)
            draw_text(frame, f"Shapes: {len(shapes)}", (15, 55), 0.7, WHITE)
            draw_text(frame, f"FPS: {int(fps)}", (15, 85), 0.7, GREEN)
            if fullscreen:
                draw_text(frame, "Vollbild: AN", (15, 115), 0.6, BLUE)
            else:
                draw_text(frame, "Vollbild: AUS", (15, 115), 0.6, BLUE)

            if show_help:
                draw_text(frame, "Q/ESC Ende   F Vollbild   H Hilfe   C Reset", (15, h - 55), 0.55, WHITE)
                draw_text(frame, "Faust = Menü   Zeigefinger = Pointer   Daumen+Zeigefinger = Auswahl", (15, h - 22), 0.55, WHITE)

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

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
