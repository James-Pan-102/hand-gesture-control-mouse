# VR Hand Gesture Control

> Control your mouse and keyboard with nothing but hand gestures — tracked in real time from your webcam with a glowing neon skeleton overlay.

[![Python](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.13-blue?logo=python&logoColor=white)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/mediapipe-0.10%2B-brightgreen?logo=mediapipe)](https://mediapipe.dev)
[![Platform](https://img.shields.io/badge/platform-windows-lightgrey?logo=windows)](https://windows.com)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## Features

- **Touchless control** — move the cursor, click, drag, and scroll without touching a thing
- **Neon HUD overlay** — real-time hand skeleton with glow effects, gesture name, FPS, and cursor position
- **4 gesture modes** — Point, Pinch, Peace, Palm
- **Smooth cursor** — rolling average + easing for jitter-free tracking
- **Pause & toggle** — keyboard shortcuts to pause control or toggle the camera mid-session

---

## Getting Started

### Prerequisites

- Python **3.9 – 3.13** ([download](https://www.python.org/downloads/))
- A webcam (built-in or USB)

### One-click launch

Just double-click **`setup_and_run.bat`** — it installs dependencies and starts the app.

### Manual setup

```bash
pip install -r requirements.txt
python hand_control.py
```

---

## Gesture Reference

| Gesture | Hand Shape | Action |
|---|---|---|
| **POINT** | Index finger up | Move cursor |
| **PINCH** | Thumb + index close | Left click |
| **PEACE** | Index + middle up | Right click |
| **PALM** | All fingers open, tilt up/down | Scroll |

### Overlay window shortcuts

| Key | Action |
|---|---|
| `Q` / `Esc` | Quit |
| `Space` | Pause / resume gesture control |
| `C` | Toggle camera on / off |

---

## Configuration

Open `hand_control.py` and adjust the constants at the top of the file:

| Constant | Default | Description |
|---|---|---|
| `CAM_INDEX` | `1` | Camera device index (`0` = external, `1` = built-in) |
| `CAM_WIDTH` | `640` | Capture resolution width (lower = faster) |
| `CAM_HEIGHT` | `480` | Capture resolution height (lower = faster) |
| `SMOOTHING` | `7` | Rolling average window (higher = smoother but laggier) |
| `MOVE_SPEED` | `1.2` | Cursor speed multiplier |
| `PINCH_THRESHOLD` | `0.045` | Normalized distance to trigger pinch (lower = tighter) |
| `CLICK_HOLD_MS` | `120` | Minimum ms between repeated clicks |

---

## How it works

1. **MediaPipe Hand Landmarker** detects 21 hand landmarks from each webcam frame
2. A **gesture classifier** reads finger positions (tip vs. PIP joint) to determine the pose
3. The **GestureController** maps the index finger position to screen coordinates and executes the appropriate action via `pyautogui`
4. A **glowing skeleton** is drawn onto the frame using OpenCV with layered glow + crisp bone rendering

```
Webcam frame → MediaPipe inference → 21 landmarks
                    ↓
            Gesture classifier  →  "POINT", "PINCH", etc.
                    ↓
            GestureController  →  pyautogui.moveTo / click / scroll
                    ↓
         OpenCV overlay  →  Floating preview window
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| **"Camera not found"** | Change `CAM_INDEX` to `0`, `1`, or `2` |
| **Jittery cursor** | Increase `SMOOTHING` (try `12`–`15`) |
| **Too fast / slow** | Adjust `MOVE_SPEED` (try `0.8` or `1.8`) |
| **Laggy detection** | Lower `CAM_WIDTH` / `CAM_HEIGHT` (try `320`×`240`) |
| **Pinch misfires** | Lower `PINCH_THRESHOLD` (try `0.03`) |
| **Poor tracking** | Improve lighting, unclutter background, keep hand 30–60 cm away |

---

## Project structure

```
hand-gesture-control/
├── hand_control.py       # Main application
├── requirements.txt      # Python dependencies
├── setup_and_run.bat     # One-click Windows launcher
├── hand_landmarker.task  # MediaPipe model file
├── README.md
└── LICENSE
```

---

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
