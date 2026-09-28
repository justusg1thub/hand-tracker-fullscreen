"""
Hand Tracker mit verbesserter Gestenerkennung und Maussteuerung

Installation:
    pip install opencv-python mediapipe pyautogui

Start:
    python hand_tracker_gesteuert.py

Tasten:
    Q / ESC     Programm beenden
    M           Mausteuerung an/aus
    F           Vollbild an/aus
    L           Skelettlinien an/aus
    T           Fingerspur an/aus
    B           Bounding Box an/aus
    H           Hilfe an/aus
    C           Spur löschen
    S           Screenshot speichern

Gesten:
    Zeigefinger              Maus bewegen
    Daumen + Zeigefinger     Linksklick
    Daumen + Mittelfinger    Rechtsklick
    Offene Hand              Maus pausieren
    Faust                    Maus wieder aktivieren
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

try:
    import pyautogui

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.01
    HAS_PYAUTOGUI = True
except Exception:
    HAS_PYAUTOGUI = False

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task",
)

WINDOW_NAME = "Hand Tracker - Gestensteuerung"

GREEN = (0, 255, 0)
DARK_GREEN = (0, 150, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 100, 0)
CYAN = (255, 255, 0)

# Hand Landmarks
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


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return

    print("Lade Hand-Modell herunter ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modell geladen.")
    except Exception as error:
        raise SystemExit(
            f"Download fehlgeschlagen: {error}\n"
            f"Lade manuell: {MODEL_URL}\n"
            f"und lege es hier ab: {MODEL_PATH}"
        )


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def landmarks_to_pixels(landmarks, width, height):
    return [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]


def angle_between(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    len_ba = math.hypot(*ba)
    len_bc = math.hypot(*bc)
    if len_ba == 0 or len_bc == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (len_ba * len_bc)))
    return math.degrees(math.acos(cos_angle))


def is_finger_extended(points, mcp, pip, tip):
    angle = angle_between(points[mcp], points[pip], points[tip])
    palm_size = distance(points[WRIST], points[MIDDLE_MCP]) + 1e-6
    tip_to_wrist = distance(points[tip], points[WRIST])
    pip_to_wrist = distance(points[pip], points[WRIST])

    return (
        angle > 145
        and tip_to_wrist > pip_to_wrist * 1.08
        and tip_to_wrist > palm_size * 0.65
    )


def is_thumb_extended(points):
    palm_size = distance(points[WRIST], points[MIDDLE_MCP]) + 1e-6
    tip_to_wrist = distance(points[THUMB_TIP], points[WRIST])
    ip_to_wrist = distance(points[THUMB_IP], points[WRIST])
    return tip_to_wrist > ip_to_wrist * 1.12 and tip_to_wrist > palm_size * 0.55


def fingers_up(points):
    thumb = is_thumb_extended(points)
    index = is_finger_extended(points, 5, INDEX_PIP, INDEX_TIP)
    middle = is_finger_extended(points, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP)
    ring = is_finger_extended(points, 13, RING_PIP, RING_TIP)
    pinky = is_finger_extended(points, 17, PINKY_PIP, PINKY_TIP)
    return [thumb, index, middle, ring, pinky]


def pinch_ratio(points, finger_tip):
    palm = distance(points[WRIST], points[MIDDLE_MCP]) + 1e-6
    return distance(points[THUMB_TIP], points[finger_tip]) / palm


def detect_gesture(points, fingers):
    thumb, index, middle, ring, pinky = fingers
    finger_count = sum(fingers)

    if not thumb and not index and not middle and not ring and not pinky:
        return "Faust"

    if finger_count == 5:
        return "Offene Hand"

    index_pinch = pinch_ratio(points, INDEX_TIP)
    middle_pinch = pinch_ratio(points, MIDDLE_TIP)

    if index_pinch < 0.30 and (thumb or index):
        return "Linksklick"

    if middle_pinch < 0.30 and (thumb or middle):
        return "Rechtsklick"

    if index and not middle and not ring and not pinky:
        return "Zeigen"

    if index and middle and not ring and not pinky:
        return "Zwei Finger"

    if thumb and not index and not middle and not ring and not pinky:
        return "Daumen hoch" if points[THUMB_TIP][1] < points[WRIST][1] else "Daumen runter"

    return f"{finger_count} Finger"


def make_bounding_box(points, width, height, padding=20):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (
        max(min(xs) - padding, 0),
        max(min(ys) - padding, 0),
        min(max(xs) + padding, width - 1),
        min(max(ys) + padding, height - 1),
    )


def draw_text(image, text, position, scale=0.7, color=WHITE, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    text_w, text_h = text_size
    cv2.rectangle(image, (x - 5, y - text_h - 8), (x + text_w + 5, y + baseline + 3), BLACK, -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_hand(image, points, show_lines=True):
    if show_lines:
        for a, b in HAND_CONNECTIONS:
            cv2.line(image, points[a], points[b], DARK_GREEN, 2, cv2.LINE_AA)
    for i, p in enumerate(points):
        r = 9 if i in (4, 8, 12, 16, 20) else 6
        cv2.circle(image, p, r, GREEN, -1, cv2.LINE_AA)
        cv2.circle(image, p, r, BLACK, 1, cv2.LINE_AA)


def draw_trail(image, trail):
    pts = list(trail)
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        color = (0, int(255 * alpha), int(80 * (1 - alpha)))
        cv2.line(image, pts[i - 1], pts[i], color, max(1, int(alpha * 8)), cv2.LINE_AA)


def draw_pinch_line(image, points, finger_tip, color):
    cv2.line(image, points[THUMB_TIP], points[finger_tip], color, 3, cv2.LINE_AA)


class MouseController:
    def __init__(self):
        self.enabled = False
        self.paused_by_open_hand = False
        self.previous_mouse_position = None
        self.previous_finger_position = None
        self.left_button_down = False
        self.right_button_down = False
        self.smoothing = 0.15
        self.acceleration = 1.2
        self.screen_width = pyautogui.size()[0] if HAS_PYAUTOGUI else 0
        self.screen_height = pyautogui.size()[1] if HAS_PYAUTOGUI else 0

    def toggle(self):
        if not HAS_PYAUTOGUI:
            print("pyautogui fehlt: pip install pyautogui")
            return
        self.enabled = not self.enabled
        self.paused_by_open_hand = False
        self.previous_mouse_position = None
        self.previous_finger_position = None
        self.release_buttons()
        print("Mausteuerung aktiviert." if self.enabled else "Mausteuerung deaktiviert.")

    def release_buttons(self):
        if not HAS_PYAUTOGUI:
            return
        if self.left_button_down:
            pyautogui.mouseUp(button="left")
            self.left_button_down = False
        if self.right_button_down:
            pyautogui.mouseUp(button="right")
            self.right_button_down = False

    def pause(self):
        self.paused_by_open_hand = True
        self.previous_mouse_position = None
        self.previous_finger_position = None
        self.release_buttons()

    def resume(self):
        self.paused_by_open_hand = False
        self.previous_mouse_position = None
        self.previous_finger_position = None

    def move_mouse(self, index_tip, frame_width, frame_height):
        if not self.enabled or self.paused_by_open_hand:
            return

        margin = 0.12
        nx = (index_tip[0] / frame_width - margin) / (1 - 2 * margin)
        ny = (index_tip[1] / frame_height - margin) / (1 - 2 * margin)
        nx = max(0.0, min(nx, 1.0))
        ny = max(0.0, min(ny, 1.0))

        target_x = nx * self.screen_width
        target_y = ny * self.screen_height

        if self.previous_mouse_position is None:
            self.previous_mouse_position = (target_x, target_y)
            self.previous_finger_position = index_tip

        if self.previous_finger_position is not None:
            finger_delta = distance(index_tip, self.previous_finger_position)
            acceleration = min(1.0 + finger_delta / 100.0, self.acceleration) if finger_delta > 5 else 1.0
        else:
            acceleration = 1.0

        self.previous_finger_position = index_tip
        prev_x, prev_y = self.previous_mouse_position
        smooth_x = prev_x + (target_x - prev_x) * self.smoothing * acceleration
        smooth_y = prev_y + (target_y - prev_y) * self.smoothing * acceleration
        self.previous_mouse_position = (smooth_x, smooth_y)
        pyautogui.moveTo(int(smooth_x), int(smooth_y), duration=0)

    def update_clicks(self, points):
        if not self.enabled or self.paused_by_open_hand:
            return

        fingers = fingers_up(points)
        thumb, index, middle, ring, pinky = fingers
        index_ratio = pinch_ratio(points, INDEX_TIP)
        middle_ratio = pinch_ratio(points, MIDDLE_TIP)

        left_pinch = index_ratio < 0.30 and (thumb or index)
        right_pinch = middle_ratio < 0.30 and (thumb or middle)

        if left_pinch and not self.left_button_down:
            pyautogui.click(button="left")
            self.left_button_down = True
        elif index_ratio > 0.42:
            self.left_button_down = False

        if right_pinch and not self.right_button_down:
            pyautogui.click(button="right")
            self.right_button_down = True
        elif middle_ratio > 0.42:
            self.right_button_down = False

    def close(self):
        self.release_buttons()


def main():
    ensure_model()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Kamera konnte nicht geöffnet werden. Probiere --camera 1")
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

    show_lines = True
    show_trail = True
    show_box = True
    show_help = True
    fullscreen = True
    mouse = MouseController()
    trails = {0: deque(maxlen=60), 1: deque(maxlen=60)}

    start = time.time()
    prev_time = start
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
            seen = set()

            for idx, lms in enumerate(result.hand_landmarks):
                seen.add(idx)
                pts = landmarks_to_pixels(lms, w, h)

                label, score = "Right", 0.0
                if idx < len(result.handedness) and result.handedness[idx]:
                    cat = result.handedness[idx][0]
                    label, score = cat.category_name, cat.score

                fingers = fingers_up(pts)
                gesture = detect_gesture(pts, fingers)

                draw_hand(frame, pts, show_lines)

                if show_box:
                    x1, y1, x2, y2 = make_bounding_box(pts, w, h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                    name = "Rechte Hand" if label == "Right" else "Linke Hand"
                    draw_text(frame, f"{name} ({int(score * 100)}%)", (x1, max(y1 - 10, 20)), 0.6, GREEN)
                    draw_text(frame, f"{gesture} | Finger: {sum(fingers)}", (x1, min(y2 + 25, h - 10)), 0.6, WHITE)

                draw_pinch_line(frame, pts, INDEX_TIP, YELLOW)

                if idx in trails:
                    trails[idx].append(pts[INDEX_TIP])
                    if show_trail:
                        draw_trail(frame, trails[idx])

                if idx == 0:
                    mouse.move_mouse(pts[INDEX_TIP], w, h)
                    mouse.update_clicks(pts)
                    if gesture == "Offene Hand":
                        mouse.pause()
                    elif gesture == "Faust":
                        mouse.resume()

            for idx in trails:
                if idx not in seen:
                    trails[idx].clear()

            now = time.time()
            inst = 1.0 / max(now - prev_time, 1e-6)
            fps = fps * 0.9 + inst * 0.1 if fps else inst
            prev_time = now

            draw_text(frame, f"FPS: {int(fps)}", (15, 30), 0.7, GREEN)
            draw_text(frame, f"Haende: {len(seen)}", (15, 60), 0.7, GREEN)
            if mouse.enabled and not mouse.paused_by_open_hand:
                draw_text(frame, "MAUS AKTIV", (15, 90), 0.7, RED)
            elif mouse.enabled and mouse.paused_by_open_hand:
                draw_text(frame, "MAUS PAUSIERT", (15, 90), 0.7, YELLOW)
            draw_text(frame, "Vollbild: AN" if fullscreen else "Vollbild: AUS", (15, 120), 0.6, BLUE)

            if show_help:
                draw_text(frame, "Q/ESC Ende  M: Maus  F: Vollbild  H: Hilfe", (15, h - 75), 0.55)
                draw_text(frame, "Zeigefinger: Maus  Daumen+Zeigefinger: Linksklick", (15, h - 45), 0.55)
                draw_text(frame, "Daumen+Mittelfinger: Rechtsklick  Hand offen: Pause", (15, h - 15), 0.55)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("m"):
                mouse.toggle()
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == ord("l"):
                show_lines = not show_lines
            elif key == ord("t"):
                show_trail = not show_trail
            elif key == ord("b"):
                show_box = not show_box
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("c"):
                for t in trails.values():
                    t.clear()
            elif key == ord("s"):
                name = datetime.now().strftime("hand_%Y%m%d_%H%M%S.png")
                cv2.imwrite(name, frame)
                print(f"Screenshot gespeichert: {name}")
    finally:
        mouse.close()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
