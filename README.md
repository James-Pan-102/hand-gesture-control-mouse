# VR Hand Gesture Control — Windows Edition

Controls your mouse and keyboard using hand gestures detected from your webcam.
Displays a glowing skeleton overlay in a floating window.

![Demo](docs/demo.gif)

## Quick Start

1. Make sure **Python 3.9+** is installed ([python.org](https://www.python.org/downloads/))
2. Double-click `setup_and_run.bat` (first run installs packages — takes ~1 minute)
3. Point your hand at the webcam and go!

### Manual install

```bash
pip install -r requirements.txt
python hand_control.py
```

## Gesture Reference

| Gesture                         | Action              |
| ------------------------------- | ------------------- |
| ☝️ POINT  (index up)           | Move mouse cursor   |
| 🤏 PINCH  (thumb + index close) | Left click          |
| ✌️ PEACE  (index + middle up)  | Right click         |
| ✊ FIST   (all curled)         | Click & hold / drag |
| 🖐️ PALM   (all open)           | Scroll (tilt hand)  |
| 🤟 THREE  (index+middle+ring)  | Middle click / paste |

**Overlay window keys:** `Space` → Pause/Resume &nbsp; `C` → Toggle camera &nbsp; `Q` → Quit

## Configuration

Open `hand_control.py` and tweak the settings at the top:

- `CAM_INDEX` — change camera (0 = external, 1 = built-in)
- `SMOOTHING` — higher = smoother but laggier (default: 7)
- `MOVE_SPEED` — cursor speed multiplier (default: 1.2)
- `PINCH_THRESHOLD` — lower = needs closer pinch (default: 0.045)
- `CAM_WIDTH` / `CAM_HEIGHT` — lower = faster but more pixelated

## Tips

- Use in a **well-lit room** for best tracking
- Keep your hand **30–60 cm** from the camera
- If jittery, increase `SMOOTHING` (try 12–15)
- Wrong camera? Change `CAM_INDEX` to 0, 1, or 2

## Files

```
hand_control.py       Main application
requirements.txt      Python dependencies
setup_and_run.bat     One-click setup + launch (Windows)
```

## License

MIT
