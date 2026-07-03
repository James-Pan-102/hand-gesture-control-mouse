"""
VR Hand Gesture Control for Windows
------------------------------------
Controls your mouse and keyboard using hand gestures detected from your webcam.
Displays a glowing skeleton overlay in a floating window.

Gestures:
  Fist (all fingers closed)        → Move mouse cursor
  OK sign (thumb-index circle)     → Click (right hand = left click, left hand = right click)
  Open Palm (all up, swipe)        → Scroll (swipe up/down)

Press Q in the overlay window to quit.
Press Space to toggle gesture control on/off (pause mode).
Press C to toggle the camera on/off.
"""

import cv2
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

OK_THRESHOLD    = 0.05       # Normalized distance for thumb-index circle (OK sign)
SCROLL_SPEED    = 40         # Scroll speed multiplier (delta from origin → scroll amount)

# Gesture-to-click mapping — change these to swap which hand does which click
RIGHT_HAND_OK_ACTION = "left_click"   # "left_click" or "right_click"
LEFT_HAND_OK_ACTION  = "right_click"  # "left_click" or "right_click"

# Glowing skeleton colors (BGR)
COLOR_BONE      = (0, 200, 255)   # Cyan-ish
COLOR_JOINT     = (255, 180, 0)   # Orange-gold
COLOR_GLOW      = (0, 100, 180)   # Deep blue glow
COLOR_TIP       = (0, 255, 120)   # Green tips
HUD_COLOR       = (0, 255, 255)   # Yellow HUD text
HUD_BG          = (20, 20, 40)    # Dark HUD background

# Hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

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

    # FIST: index, middle, ring, pinky all closed
    if not up[1] and not up[2] and not up[3] and not up[4]:
        return "FIST"

    # OK: thumb-index touching + at least 2 of middle/ring/pinky extended
    ok_dist = math.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y)
    if ok_dist < OK_THRESHOLD and sum(up[2:]) >= 2:
        return "OK"

    # PALM: 4+ fingers up (open hand for swipe)
    if sum(up) >= 4:
        return "PALM"

    return "IDLE"


# ── Glowing draw functions ─────────────────────────────────────────────────────

def draw_glowing_skeleton(frame, lms_px):
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

    def draw_panel(px, py, pw, ph, title, lines):
        panel = frame[py:py+ph, px:px+pw].copy()
        dark = np.full_like(panel, HUD_BG)
        cv2.addWeighted(dark, 0.75, panel, 0.25, 0, panel)
        frame[py:py+ph, px:px+pw] = panel
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), HUD_COLOR, 1)
        cv2.rectangle(frame, (px, py), (px+pw, py+22), (40, 40, 80), -1)
        cv2.putText(frame, f"  {title}", (px+4, py+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_COLOR, 1)
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (px+10, py+28+i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Left panel — status
    g_colors = {
        "FIST" : (0, 255, 255), "OK" : (0, 200, 255),
        "PALM" : (0, 255, 180), "IDLE" : (120, 120, 120),
    }
    first_gesture = gesture.split()[0] if gesture else "IDLE"
    first_gesture = first_gesture.split(":")[-1] if ":" in first_gesture else first_gesture
    g_color = g_colors.get(first_gesture, HUD_COLOR)
    cam_label = "CAM ON" if camera_on else "CAM OFF"
    cam_color = (0, 255, 200) if camera_on else (0, 80, 255)
    ctrl_label = "ACTIVE" if active else "PAUSED"
    ctrl_color = (0, 255, 80) if active else (0, 80, 255)

    draw_panel(10, 10, 300, 190, "VR HAND CONTROL", [
        (f"CAMERA  : {cam_label}", cam_color),
        (f"CONTROL : {ctrl_label}", ctrl_color),
        (f"GESTURE : {gesture}", g_color),
        (f"CURSOR  : {screen_pos[0]}, {screen_pos[1]}", HUD_COLOR),
        (f"FPS     : {fps:.0f}", HUD_COLOR),
        ("[C]=Camera  [Space]=Pause  [Q]=Quit", (160, 160, 200)),
    ])

    # Bottom panel — gesture instructions
    r_act = "Left" if RIGHT_HAND_OK_ACTION == "left_click" else "Right"
    l_act = "Right" if LEFT_HAND_OK_ACTION == "right_click" else "Left"
    inst_lines = [
        ("FIST: all closed          -> Move cursor",      (0, 255, 255)),
        (f"OK R-hand: circle         -> {r_act} click",     (0, 200, 255)),
        (f"OK L-hand: circle         -> {l_act} click",     (0, 180, 255)),
        ("PALM: all open + swipe     -> Scroll",           (0, 255, 180)),
    ]
    iw = w - 20
    ih = 22 + len(inst_lines) * 18 + 6
    ix = 10
    iy = h - ih - 10
    draw_panel(ix, iy, iw, ih, "GESTURES", inst_lines)


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
        nx = np.clip((lm.x - margin_x) / (1 - 2 * margin_x), 0, 1)
        ny = np.clip((lm.y - margin_y) / (1 - 2 * margin_y), 0, 1)
        return int(nx * self.sw), int(ny * self.sh)

    def smooth_move(self, tx, ty):
        self.smooth_x.append(tx)
        self.smooth_y.append(ty)
        sx = int(np.mean(self.smooth_x))
        sy = int(np.mean(self.smooth_y))
        cx, cy = pyautogui.position()
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

    def handle(self, gesture, lms, frame_w, frame_h, handedness="Unknown"):
        cur_pos = pyautogui.position()

        if gesture == "FIST":
            index_tip = lms[8]
            tx, ty = self.map_to_screen(index_tip, frame_w, frame_h)
            self.smooth_move(tx, ty)
            cur_pos = (tx, ty)
            self.scroll_origin = None
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False

        elif gesture == "OK":
            self.scroll_origin = None
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False

            if self.can_click():
                if handedness == "Right":
                    if RIGHT_HAND_OK_ACTION == "left_click":
                        pyautogui.click(button='left')
                    else:
                        pyautogui.click(button='right')
                elif handedness == "Left":
                    if LEFT_HAND_OK_ACTION == "right_click":
                        pyautogui.click(button='right')
                    else:
                        pyautogui.click(button='left')

        elif gesture == "PALM":
            wrist_y = lms[0].y
            if self.scroll_origin is None:
                self.scroll_origin = wrist_y
            delta = (self.scroll_origin - wrist_y) * SCROLL_SPEED
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
            num_hands=2,
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
                gesture_parts = []
                for i, lms in enumerate(result.hand_landmarks):
                    lms_px = [landmark_px(lm, w, h) for lm in lms]
                    draw_glowing_skeleton(frame, lms_px)
                    g = classify_gesture(lms)

                    hand_name = "?"
                    if result.handedness and i < len(result.handedness):
                        hand_name = result.handedness[i][0].category_name[0]
                    gesture_parts.append(f"{hand_name}:{g}")

                    if active:
                        handedness = result.handedness[i][0].category_name if result.handedness else "Unknown"
                        cur_pos = controller.handle(g, lms, w, h, handedness)
                gesture = " ".join(gesture_parts)
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
