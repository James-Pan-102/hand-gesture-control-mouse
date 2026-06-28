"""
VR Hand Gesture Control for Windows
------------------------------------
Controls your mouse and keyboard using hand gestures detected from your webcam.
Displays a glowing skeleton overlay in a floating window.

Gestures:
  Point (index up)          → Move mouse cursor
  Pinch (thumb + index)     → Left click
  Peace (index + middle up) → Right click
  Fist (all curled)         → Click & hold / drag
  Open Palm (all up)        → Scroll mode (tilt to scroll)
  Three fingers up          → Middle click / paste (Ctrl+V)

Press Q in the overlay window to quit.
Press Space to toggle gesture control on/off (pause mode).
Press C to toggle the camera on/off.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
import pyautogui
import numpy as np
import time
import math
from collections import deque

HOLD_MS = 500  # hold a gesture this long to trigger secondary action

# ── Configuration ─────────────────────────────────────────────────────────────

CAM_INDEX       = 1          # 0 = external cam, 1 = built-in webcam
CAM_WIDTH       = 640
CAM_HEIGHT      = 480
FLIP_HORIZONTAL = True       # Mirror the camera so it feels natural

SMOOTHING       = 7          # Rolling average frames for mouse smoothing (higher = smoother)
MOVE_SPEED      = 1.2        # Mouse movement speed multiplier
CLICK_HOLD_MS   = 120        # Min milliseconds between repeated clicks
PINCH_THRESHOLD = 0.045      # Distance (normalized) to trigger pinch
FIST_THRESHOLD  = 0.06       # Distance for fist detection

# Glowing skeleton colors (BGR)
COLOR_BONE      = (0, 200, 255)   # Cyan-ish
COLOR_JOINT     = (255, 180, 0)   # Orange-gold
COLOR_GLOW      = (0, 100, 180)   # Deep blue glow
COLOR_TIP       = (0, 255, 120)   # Green tips
HUD_COLOR       = (0, 255, 255)   # Yellow HUD text
HUD_BG          = (20, 20, 40)    # Dark HUD background

# Hand landmark connections (replaces removed mp.solutions.hands.HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def dist(a, b):
    """Euclidean distance between two landmarks (normalized coords)."""
    return math.hypot(a.x - b.x, a.y - b.y)


def landmark_px(lm, w, h):
    """Convert normalized landmark to pixel coords."""
    return int(lm.x * w), int(lm.y * h)


def fingers_up(lms):
    """
    Returns a list of booleans [thumb, index, middle, ring, pinky]
    indicating which fingers are extended.
    """
    tips  = [4, 8, 12, 16, 20]
    mcps  = [2, 6, 10, 14, 18]
    up    = []

    # Thumb: compare x (left/right) rather than y
    if lms[4].x < lms[3].x:   # right hand mirrored
        up.append(True)
    else:
        up.append(False)

    # Other four fingers: tip y < pip y means extended
    for tip, mcp in zip(tips[1:], mcps[1:]):
        up.append(lms[tip].y < lms[tip - 2].y)

    return up


def classify_gesture(lms):
    up = fingers_up(lms)
    n_up = sum(up)

    if n_up == 0 or (n_up == 1 and up[0]):
        return "FIST"

    if n_up >= 4:
        return "PALM"

    if up[1] and up[2] and not up[3] and not up[4]:
        return "PEACE"

    if up[1] and not up[2] and not up[3] and not up[4]:
        return "POINT"

    return "IDLE"


# ── Glowing draw functions ─────────────────────────────────────────────────────

def draw_glowing_skeleton(frame, lms_px, gesture):
    """Draw a neon-glowing hand skeleton on the frame."""
    overlay = frame.copy()

    # Draw glow (thick, blurred)
    for conn in HAND_CONNECTIONS:
        a, b = conn
        pt1 = lms_px[a]
        pt2 = lms_px[b]
        cv2.line(overlay, pt1, pt2, COLOR_GLOW, 12)

    # Blend glow
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Draw crisp bones on top
    for conn in HAND_CONNECTIONS:
        a, b = conn
        cv2.line(frame, lms_px[a], lms_px[b], COLOR_BONE, 2)

    # Draw joints
    tip_ids = {4, 8, 12, 16, 20}
    for i, pt in enumerate(lms_px):
        color = COLOR_TIP if i in tip_ids else COLOR_JOINT
        radius = 7 if i in tip_ids else 5
        cv2.circle(frame, pt, radius + 4, COLOR_GLOW, -1)   # glow ring
        cv2.circle(frame, pt, radius, color, -1)             # bright center


def draw_hud(frame, gesture, active, camera_on, fps, screen_pos):
    """Draw the Windows-style HUD overlay."""
    h, w = frame.shape[:2]

    # HUD panel
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 320, 190
    panel = frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w].copy()
    dark = np.full_like(panel, HUD_BG)
    cv2.addWeighted(dark, 0.75, panel, 0.25, 0, panel)
    frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w] = panel

    # Border
    cv2.rectangle(frame,
                  (panel_x, panel_y),
                  (panel_x+panel_w, panel_y+panel_h),
                  HUD_COLOR, 1)

    # Title bar strip
    cv2.rectangle(frame,
                  (panel_x, panel_y),
                  (panel_x+panel_w, panel_y+22),
                  (40, 40, 80), -1)
    cv2.putText(frame, "  VR HAND CONTROL",
                (panel_x+4, panel_y+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_COLOR, 1)

    # Status
    cam_label = "CAM ON" if camera_on else "CAM OFF"
    cam_color = (0, 255, 200) if camera_on else (0, 80, 255)
    ctrl_label = "ACTIVE" if active else "PAUSED"
    ctrl_color = (0, 255, 80) if active else (0, 80, 255)
    cv2.putText(frame, f"CAMERA  : {cam_label}",
                (panel_x+10, panel_y+44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cam_color, 1)
    cv2.putText(frame, f"CONTROL : {ctrl_label}",
                (panel_x+10, panel_y+68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ctrl_color, 1)

    # Gesture
    g_colors = {
        "PEACE" : (0, 180, 255),
        "FIST"  : (0, 80, 255),
        "PALM"  : (0, 255, 180),
        "POINT" : (0, 255, 255),
        "IDLE"  : (120, 120, 120),
    }
    g_color = g_colors.get(gesture, HUD_COLOR)
    cv2.putText(frame, f"GESTURE: {gesture}",
                (panel_x+10, panel_y+92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, g_color, 1)

    # Cursor position
    cv2.putText(frame, f"CURSOR : {screen_pos[0]}, {screen_pos[1]}",
                (panel_x+10, panel_y+116),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_COLOR, 1)

    # FPS
    cv2.putText(frame, f"FPS    : {fps:.0f}",
                (panel_x+10, panel_y+140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_COLOR, 1)

    # Controls hint
    cv2.putText(frame, "  [C]=Camera  [Space]=Pause  [Q]=Quit",
                (panel_x+4, panel_y+164),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 200), 1)


# ── Gesture action handler ─────────────────────────────────────────────────────

class GestureController:
    def __init__(self):
        self.sw, self.sh = pyautogui.size()
        self.smooth_x = deque(maxlen=SMOOTHING)
        self.smooth_y = deque(maxlen=SMOOTHING)
        self.last_click_time = 0
        self.dragging       = False
        self.prev_gesture   = "IDLE"
        self.gesture_start  = 0
        self.scroll_origin  = None

    def map_to_screen(self, lm, frame_w, frame_h):
        """Map a landmark's position to screen coordinates with margins."""
        margin_x = 0.10
        margin_y = 0.10
        # Clamp and remap [margin, 1-margin] → [0, screen]
        nx = np.clip((lm.x - margin_x) / (1 - 2 * margin_x), 0, 1)
        ny = np.clip((lm.y - margin_y) / (1 - 2 * margin_y), 0, 1)
        return int(nx * self.sw), int(ny * self.sh)

    def smooth_move(self, tx, ty):
        self.smooth_x.append(tx)
        self.smooth_y.append(ty)
        sx = int(np.mean(self.smooth_x))
        sy = int(np.mean(self.smooth_y))
        cx, cy = pyautogui.position()
        # Ease toward target
        nx = int(cx + (sx - cx) * MOVE_SPEED)
        ny = int(cy + (sy - cy) * MOVE_SPEED)
        pyautogui.moveTo(nx, ny)
        return nx, ny

    def can_click(self):
        now = time.time() * 1000
        if now - self.last_click_time > CLICK_HOLD_MS:
            self.last_click_time = now
            return True
        return False

    def handle(self, gesture, lms, frame_w, frame_h):
        index_tip = lms[8]
        tx, ty    = self.map_to_screen(index_tip, frame_w, frame_h)
        cur_pos   = pyautogui.position()
        now_ms    = time.time() * 1000

        if gesture == "POINT":
            self.smooth_move(tx, ty)
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False

        elif gesture == "PEACE":
            self.smooth_move(tx, ty)
            if self.prev_gesture != "PEACE":
                self.gesture_start = now_ms
                if self.can_click():
                    pyautogui.click()
            elif now_ms - self.gesture_start > HOLD_MS:
                if not self.dragging:
                    pyautogui.mouseDown()
                    self.dragging = True

        elif gesture == "FIST":
            self.smooth_move(tx, ty)
            if self.prev_gesture != "FIST" and self.can_click():
                pyautogui.rightClick()
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False

        elif gesture == "PALM":
            wrist_y = lms[0].y
            if self.scroll_origin is None:
                self.scroll_origin = wrist_y
            delta = (self.scroll_origin - wrist_y) * 30
            if abs(delta) > 0.5:
                pyautogui.scroll(int(delta))
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False

        else:
            self.scroll_origin = None
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False

        if gesture != "PALM":
            self.scroll_origin = None

        self.prev_gesture = gesture
        return cur_pos


# ── Main loop ──────────────────────────────────────────────────────────────────

def init_camera():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        time.sleep(0.5)
        cap.grab()
    return cap


def main():
    print("=" * 60, flush=True)
    print("  VR Hand Gesture Control  |  Windows Edition", flush=True)
    print("=" * 60, flush=True)

    cap = init_camera()
    if not cap.isOpened():
        print(f"ERROR: Camera {CAM_INDEX} not available. Try changing CAM_INDEX.")
        return

    print(f"  Camera {CAM_INDEX} opened. Press [C] to toggle camera, [Space] to pause.")
    print()

    pyautogui.FAILSAFE = False

    controller = GestureController()
    active      = True
    fps_buf     = deque(maxlen=30)
    prev_time   = time.time()
    cur_pos     = (0, 0)
    gesture     = "IDLE"

    cv2.namedWindow("VR Hand Control", cv2.WINDOW_NORMAL)

    hand_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.6,
        )
    )

    camera_on   = True
    frame_failures = 0

    while True:
        if camera_on:
            ret, frame = cap.read()
            if not ret:
                frame_failures += 1
                if frame_failures > 30:
                    print("Camera read failed repeatedly — exiting.")
                    break
                time.sleep(0.1)
                continue
            frame_failures = 0

            h, w = frame.shape[:2]

            if FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            # FPS
            now      = time.time()
            fps_buf.append(1.0 / max(now - prev_time, 1e-9))
            prev_time = now
            fps      = np.mean(fps_buf)

            # Hand detection
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
            result = hand_landmarker.detect(mp_img)

            if result.hand_landmarks:
                for lms in result.hand_landmarks:
                    lms_px = [landmark_px(lm, w, h) for lm in lms]

                    draw_glowing_skeleton(frame, lms_px, gesture)
                    gesture = classify_gesture(lms)

                    if active:
                        cur_pos = controller.handle(gesture, lms, w, h)
            else:
                gesture = "IDLE"
                if controller.dragging:
                    pyautogui.mouseUp()
                    controller.dragging = False
        else:
            # Camera off — show placeholder
            frame = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
            h, w = CAM_HEIGHT, CAM_WIDTH
            fps = 0
            gesture = "IDLE"
            if controller.dragging:
                pyautogui.mouseUp()
                controller.dragging = False
            cv2.putText(frame, "CAMERA OFF — Press [C] to turn on",
                        (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        draw_hud(frame, gesture, active, camera_on, fps, cur_pos)

        cv2.imshow("VR Hand Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            active = not active
            if not active and controller.dragging:
                pyautogui.mouseUp()
                controller.dragging = False
        elif key == ord('c'):
            camera_on = not camera_on
            if camera_on:
                cap = init_camera()
                if not cap.isOpened():
                    print("Failed to re-open camera.")
                    camera_on = False
            else:
                cap.release()
                if controller.dragging:
                    pyautogui.mouseUp()
                    controller.dragging = False

    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    main()
