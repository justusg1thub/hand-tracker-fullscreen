"""
Hand Tracker mit verbesserter Gestensteuerung und Mausbewegung

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
    +/-         Empfindlichkeit anpassen

Gesten bei aktivierter Mausteuerung:
    Zeigefinger             Maus bewegen
    Daumen + Zeigefinger   Linksklick
    Daumen + Mittelfinger  Rechtsklick
    Offene Hand             Mausteuerung pausieren
    Faust                   Mausteuerung wieder aktivieren

Wichtig:
    Die Mausteuerung funktioniert nur, wenn das Tracker-Fenster sichtbar ist.
    Der Tracker startet standardmäßig im Vollbild.
"""

import math
import os
import sys
import time
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ---------------------------------------------------------------------
# PyAutoGUI laden
# ---------------------------------------------------------------------

try:
    import pyautogui

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.01
    HAS_PYAUTOGUI = True

except Exception:
    HAS_PYAUTOGUI = False


# ---------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------

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

# MediaPipe-Handpunkte
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


# ---------------------------------------------------------------------
# Modell herunterladen
# ---------------------------------------------------------------------

def ensure_model():
    """Lädt das MediaPipe-Modell beim ersten Start herunter."""

    if os.path.exists(MODEL_PATH):
        return

    print("Lade Hand-Modell herunter. Das passiert nur beim ersten Start ...")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modell erfolgreich heruntergeladen.")

    except Exception as error:
        raise SystemExit(
            f"\nModell konnte nicht heruntergeladen werden:\n{error}\n\n"
            f"Lade diese Datei manuell herunter:\n{MODEL_URL}\n\n"
            f"Speichere sie hier:\n{MODEL_PATH}"
        )


# ---------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------

def distance(point_a, point_b):
    return math.hypot(
        point_a[0] - point_b[0],
        point_a[1] - point_b[1],
    )


def landmarks_to_pixels(landmarks, width, height):
    """MediaPipe-Koordinaten in Pixel umwandeln."""

    return [
        (
            int(landmark.x * width),
            int(landmark.y * height),
        )
        for landmark in landmarks
    ]


def fingers_up(points, hand_label):
    """
    Gibt zurück, welche Finger nach oben zeigen.

    Reihenfolge:
        Daumen, Zeigefinger, Mittelfinger, Ringfinger, kleiner Finger
    """

    result = []

    # Der Daumen zeigt bei linker und rechter Hand in unterschiedliche Richtungen.
    if hand_label == "Right":
        thumb_is_up = points[THUMB_TIP][0] < points[THUMB_IP][0]
    else:
        thumb_is_up = points[THUMB_TIP][0] > points[THUMB_IP][0]

    result.append(thumb_is_up)

    for tip, pip in [
        (INDEX_TIP, INDEX_PIP),
        (MIDDLE_TIP, MIDDLE_PIP),
        (RING_TIP, RING_PIP),
        (PINKY_TIP, PINKY_PIP),
    ]:
        result.append(points[tip][1] < points[pip][1])

    return result


def pinch_ratio(points, finger_tip):
    """
    Abstand zwischen Daumen und einem Finger.
    Der Abstand wird an der Handgröße normalisiert.
    """

    hand_size = distance(points[WRIST], points[MIDDLE_MCP]) + 0.000001

    return distance(
        points[THUMB_TIP],
        points[finger_tip],
    ) / hand_size


def detect_gesture(points, fingers):
    """Erkennt einfache Gesten."""

    thumb, index, middle, ring, pinky = fingers
    finger_count = sum(fingers)

    index_pinch = pinch_ratio(points, INDEX_TIP)
    middle_pinch = pinch_ratio(points, MIDDLE_TIP)

    if index_pinch < 0.30:
        return "Linksklick"

    if middle_pinch < 0.30:
        return "Rechtsklick"

    if finger_count == 0:
        return "Faust"

    if finger_count == 5:
        return "Offene Hand"

    if index and not middle and not ring and not pinky:
        return "Zeigen"

    if index and middle and not ring and not pinky:
        return "Zwei Finger"

    if thumb and not index and not middle and not ring and not pinky:
        if points[THUMB_TIP][1] < points[WRIST][1]:
            return "Daumen hoch"
        return "Daumen runter"

    return f"{finger_count} Finger"


def make_bounding_box(points, width, height, padding=20):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    return (
        max(min(xs) - padding, 0),
        max(min(ys) - padding, 0),
        min(max(xs) + padding, width - 1),
        min(max(ys) + padding, height - 1),
    )


def draw_text(image, text, position, scale=0.7, color=WHITE, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size, baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    text_width, text_height = text_size
    x, y = position

    cv2.rectangle(
        image,
        (x - 5, y - text_height - 8),
        (x + text_width + 5, y + baseline + 3),
        BLACK,
        -1,
    )

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_hand(image, points, show_lines=True):
    if show_lines:
        for point_a, point_b in HAND_CONNECTIONS:
            cv2.line(
                image,
                points[point_a],
                points[point_b],
                DARK_GREEN,
                2,
                cv2.LINE_AA,
            )

    for index, point in enumerate(points):
        radius = 9 if index in (4, 8, 12, 16, 20) else 6

        cv2.circle(
            image,
            point,
            radius,
            GREEN,
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            image,
            point,
            radius,
            BLACK,
            1,
            cv2.LINE_AA,
        )


def draw_trail(image, trail):
    points = list(trail)

    if len(points) < 2:
        return

    for index in range(1, len(points)):
        alpha = index / len(points)

        color = (
            0,
            int(255 * alpha),
            int(80 * (1 - alpha)),
        )

        thickness = max(1, int(alpha * 8))

        cv2.line(
            image,
            points[index - 1],
            points[index],
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_pinch_line(image, points, finger_tip, color):
    cv2.line(
        image,
        points[THUMB_TIP],
        points[finger_tip],
        color,
        3,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------
# Maussteuerung mit verbesserter Kalibrierung
# ---------------------------------------------------------------------

class MouseController:
    """
    Steuert die Computermaus mit dem Zeigefinger.

    Zeigefinger:
        Maus bewegen

    Daumen + Zeigefinger:
        Linksklick

    Daumen + Mittelfinger:
        Rechtsklick
    """

    def __init__(self):
        self.enabled = False
        self.paused_by_open_hand = False

        self.previous_mouse_position = None
        self.previous_finger_position = None

        self.left_button_down = False
        self.right_button_down = False

        # Verbesserte Einstellungen für flüssigere Bewegung
        self.smoothing = 0.15  # Weniger Glättung = schnellere Reaktion
        self.acceleration = 1.2  # Beschleunigung für größere Bewegungen
        
        self.screen_width = 0
        self.screen_height = 0
        
        # Kalibrierungsbereich (wird bei aktivierung gemessen)
        self.frame_width = 0
        self.frame_height = 0

        if HAS_PYAUTOGUI:
            self.screen_width, self.screen_height = pyautogui.size()

    def toggle(self):
        if not HAS_PYAUTOGUI:
            print("PyAutoGUI ist nicht installiert.")
            print("Installation: pip install pyautogui")
            return

        self.enabled = not self.enabled
        self.paused_by_open_hand = False
        self.previous_mouse_position = None
        self.previous_finger_position = None

        self.release_buttons()

        if self.enabled:
            print("Mausteuerung aktiviert.")
        else:
            print("Mausteuerung deaktiviert.")

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

        self.frame_width = frame_width
        self.frame_height = frame_height

        # Aktivbereich reduzieren für bessere Kontrolle
        # Nur der mittlere Bereich wird für Mausbewegung genutzt
        margin = 0.10

        # Normalisierte Position im Frame
        normalized_x = (index_tip[0] / frame_width - margin) / (1 - 2 * margin)
        normalized_y = (index_tip[1] / frame_height - margin) / (1 - 2 * margin)

        # Begrenzen auf Bildschirm
        normalized_x = max(0.0, min(normalized_x, 1.0))
        normalized_y = max(0.0, min(normalized_y, 1.0))

        # Auf Bildschirmauflösung abbilden
        target_x = normalized_x * self.screen_width
        target_y = normalized_y * self.screen_height

        # Erste Position speichern
        if self.previous_mouse_position is None:
            self.previous_mouse_position = (target_x, target_y)
            self.previous_finger_position = index_tip

        # Beschleunigung basierend auf Bewegungsgeschwindigkeit
        if self.previous_finger_position:
            finger_delta = distance(
                index_tip,
                self.previous_finger_position
            )
            
            # Größere Bewegungen beschleunigen
            if finger_delta > 5:
                acceleration = min(1.0 + (finger_delta / 100), self.acceleration)
            else:
                acceleration = 1.0
        else:
            acceleration = 1.0

        self.previous_finger_position = index_tip

        # Glätten mit Beschleunigung
        previous_x, previous_y = self.previous_mouse_position

        smooth_x = previous_x + (target_x - previous_x) * self.smoothing * acceleration
        smooth_y = previous_y + (target_y - previous_y) * self.smoothing * acceleration

        self.previous_mouse_position = (smooth_x, smooth_y)

        # Maus bewegen
        pyautogui.moveTo(
            int(smooth_x),
            int(smooth_y),
            duration=0,
        )

    def update_clicks(self, points):
        if not self.enabled or self.paused_by_open_hand:
            return

        index_ratio = pinch_ratio(points, INDEX_TIP)
        middle_ratio = pinch_ratio(points, MIDDLE_TIP)

        # Linksklick: Daumen und Zeigefinger berühren sich
        if index_ratio < 0.28 and not self.left_button_down:
            pyautogui.click(button="left")
            self.left_button_down = True

        elif index_ratio > 0.42:
            self.left_button_down = False

        # Rechtsklick: Daumen und Mittelfinger berühren sich
        if middle_ratio < 0.28 and not self.right_button_down:
            pyautogui.click(button="right")
            self.right_button_down = True

        elif middle_ratio > 0.42:
            self.right_button_down = False

    def close(self):
        self.release_buttons()


# ---------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------

def main():
    ensure_model()

    # Kamera öffnen
    camera = cv2.VideoCapture(0)

    # Höhere Auflösung für bessere Fingeranalyse
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    if not camera.isOpened():
        print("Kamera konnte nicht geöffnet werden.")
        print("Probiere eine andere Kamera-Nummer, zum Beispiel 1.")
        return

    # MediaPipe-Handmodell mit besseren Einstellungen
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.50,  # Etwas niedriger für bessere Erkennung
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.50,
    )

    landmarker = vision.HandLandmarker.create_from_options(options)

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    # Direkt im Vollbild starten
    fullscreen = True

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    show_lines = True
    show_trail = True
    show_box = True
    show_help = True

    mouse = MouseController()

    trails = {
        0: deque(maxlen=60),
        1: deque(maxlen=60),
    }

    start_time = time.time()
    previous_time = start_time
    last_timestamp = -1
    fps = 0.0

    try:
        while camera.isOpened():
            success, frame = camera.read()

            if not success:
                print("Kein Bild von der Kamera.")
                break

            # Spiegelbild wie bei einer Frontkamera
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            # Bildqualität verbessern für bessere Handanalyse
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

            # OpenCV BGR zu RGB umwandeln
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # MediaPipe benötigt steigende Zeitstempel
            timestamp = int(
                (time.time() - start_time) * 1000
            )

            if timestamp <= last_timestamp:
                timestamp = last_timestamp + 1

            last_timestamp = timestamp

            result = landmarker.detect_for_video(
                mp_image,
                timestamp,
            )

            visible_hands = set()

            # Alle erkannten Hände verarbeiten
            for hand_index, landmarks in enumerate(
                result.hand_landmarks
            ):
                visible_hands.add(hand_index)

                points = landmarks_to_pixels(
                    landmarks,
                    width,
                    height,
                )

                hand_label = "Right"
                confidence = 0.0

                if (
                    hand_index < len(result.handedness)
                    and result.handedness[hand_index]
                ):
                    category = result.handedness[hand_index][0]

                    hand_label = category.category_name
                    confidence = category.score

                fingers = fingers_up(
                    points,
                    hand_label,
                )

                gesture = detect_gesture(
                    points,
                    fingers,
                )

                # Hand zeichnen
                draw_hand(
                    frame,
                    points,
                    show_lines,
                )

                # Bounding Box zeichnen
                if show_box:
                    x1, y1, x2, y2 = make_bounding_box(
                        points,
                        width,
                        height,
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        GREEN,
                        2,
                    )

                    readable_label = (
                        "Rechte Hand"
                        if hand_label == "Right"
                        else "Linke Hand"
                    )

                    draw_text(
                        frame,
                        f"{readable_label} "
                        f"({int(confidence * 100)}%)",
                        (x1, max(y1 - 10, 25)),
                        0.6,
                        GREEN,
                    )

                    draw_text(
                        frame,
                        f"{gesture} | Finger: {sum(fingers)}",
                        (x1, min(y2 + 25, height - 10)),
                        0.6,
                        WHITE,
                    )

                # Fingerspur speichern
                if hand_index in trails:
                    trails[hand_index].append(
                        points[INDEX_TIP]
                    )

                    if show_trail:
                        draw_trail(
                            frame,
                            trails[hand_index],
                        )

                # Nur die erste Hand steuert die Maus
                if hand_index == 0:
                    draw_pinch_line(
                        frame,
                        points,
                        INDEX_TIP,
                        YELLOW,
                    )

                    # Zeigefinger-Position für Maus
                    cv2.circle(
                        frame,
                        points[INDEX_TIP],
                        12,
                        CYAN,
                        2,
                        cv2.LINE_AA,
                    )

                    mouse.move_mouse(
                        points[INDEX_TIP],
                        width,
                        height,
                    )

                    mouse.update_clicks(points)

                    # Offene Hand pausiert die Mausteuerung.
                    # Eine Faust aktiviert sie wieder.
                    if gesture == "Offene Hand":
                        mouse.pause()

                    elif gesture == "Faust":
                        mouse.resume()

            # Spuren nicht mehr sichtbarer Hände löschen
            for hand_index in trails:
                if hand_index not in visible_hands:
                    trails[hand_index].clear()

            # FPS berechnen
            current_time = time.time()

            instant_fps = 1.0 / max(
                current_time - previous_time,
                0.000001,
            )

            if fps == 0:
                fps = instant_fps
            else:
                fps = fps * 0.90 + instant_fps * 0.10

            previous_time = current_time

            # Statusanzeige
            draw_text(
                frame,
                f"FPS: {int(fps)}",
                (15, 35),
                0.7,
                GREEN,
            )

            draw_text(
                frame,
                f"Haende: {len(visible_hands)}",
                (15, 70),
                0.7,
                GREEN,
            )

            if mouse.enabled and not mouse.paused_by_open_hand:
                draw_text(
                    frame,
                    "MAUS AKTIV",
                    (15, 105),
                    0.7,
                    RED,
                )

            elif mouse.enabled and mouse.paused_by_open_hand:
                draw_text(
                    frame,
                    "MAUS PAUSIERT",
                    (15, 105),
                    0.7,
                    YELLOW,
                )

            if fullscreen:
                fullscreen_text = "Vollbild: AN"
            else:
                fullscreen_text = "Vollbild: AUS"

            draw_text(
                frame,
                fullscreen_text,
                (15, 140),
                0.6,
                BLUE,
            )

            # Hilfe anzeigen
            if show_help:
                draw_text(
                    frame,
                    "Q/ESC Ende | M Maus | F Vollbild | H Hilfe",
                    (15, height - 75),
                    0.55,
                    WHITE,
                )

                draw_text(
                    frame,
                    "Zeigefinger Maus | Daumen+Zeigefinger Linksklick",
                    (15, height - 45),
                    0.55,
                    WHITE,
                )

                draw_text(
                    frame,
                    "Daumen+Mittelfinger Rechtsklick | Offene Hand Pause",
                    (15, height - 15),
                    0.55,
                    WHITE,
                )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            elif key == ord("m"):
                mouse.toggle()

            elif key == ord("f"):
                fullscreen = not fullscreen

                if fullscreen:
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                else:
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL,
                    )

            elif key == ord("l"):
                show_lines = not show_lines

            elif key == ord("t"):
                show_trail = not show_trail

            elif key == ord("b"):
                show_box = not show_box

            elif key == ord("h"):
                show_help = not show_help

            elif key == ord("c"):
                for trail in trails.values():
                    trail.clear()

            elif key == ord("s"):
                filename = datetime.now().strftime(
                    "hand_%Y%m%d_%H%M%S.png"
                )

                cv2.imwrite(filename, frame)
                print(f"Screenshot gespeichert: {filename}")

    finally:
        mouse.close()
        landmarker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
