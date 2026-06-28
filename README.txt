============================================================
  VR Hand Gesture Control  |  Windows Edition
  Glowing skeleton overlay + real mouse/keyboard control
============================================================

QUICK START
-----------
1. Make sure Python 3.9 or later is installed.
   Download: https://www.python.org/downloads/
   During install: check "Add Python to PATH" ✓

2. Double-click:  setup_and_run.bat
   (First run installs packages — takes ~1 minute)

3. Point your hand at the webcam and go!


MANUAL INSTALL (if the bat file doesn't work)
----------------------------------------------
Open Command Prompt in this folder and run:

    pip install mediapipe opencv-python pyautogui numpy
    python hand_control.py


GESTURE REFERENCE
-----------------
  POINT  (index finger up only)  →  Move mouse cursor
  PINCH  (thumb + index close)   →  Left click
  PEACE  (index + middle up)     →  Right click
  FIST   (all fingers curled)    →  Hold & drag
  PALM   (all fingers open)      →  Scroll (tilt hand up/down)
  THREE  (index+middle+ring up)  →  Middle click

  SPACE key in the overlay window  →  Pause / Resume control
  Q key in the overlay window      →  Quit


TIPS
----
- Use in a well-lit room for best tracking.
- Keep your hand ~30–60 cm from the camera.
- The overlay window shows a real-time FPS and gesture name.
- If the cursor feels jittery, increase SMOOTHING in hand_control.py.
- If the cursor moves too fast/slow, adjust MOVE_SPEED.
- Wrong camera? Change CAM_INDEX (0, 1, 2…) in hand_control.py.


TROUBLESHOOTING
---------------
"Camera not found"
  → Try changing CAM_INDEX = 1 (or 2) at the top of hand_control.py

Laggy / slow detection
  → Lower CAM_WIDTH / CAM_HEIGHT in hand_control.py
  → Change model_complexity from 1 to 0 for faster (less accurate) tracking

Mouse jumps around
  → Increase SMOOTHING value (default 7, try 12–15)
  → Make sure lighting is good and background is uncluttered

Pinch triggers too easily / not enough
  → Adjust PINCH_THRESHOLD (lower = needs closer pinch, higher = looser)


FILES
-----
  hand_control.py     Main application
  requirements.txt    Python package list
  setup_and_run.bat   One-click setup + launch (Windows)
  README.txt          This file


============================================================
