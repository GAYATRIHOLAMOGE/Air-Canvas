"""
Air Writing using OpenCV & MediaPipe
=====================================
Draw in the air using hand gestures captured via webcam.

Gesture Controls:
  - Index finger UP        → Writing mode (draw on canvas)
  - Index + Middle UP      → Cursor mode (move without drawing)
  - Fist (all fingers down)→ Clear canvas
  - Hover over color panel → Select draw color (in Cursor mode)

Keyboard Controls:
  Q / ESC  → Quit
  S        → Save canvas as PNG
  C        → Clear canvas
  + / =    → Increase brush size
  -        → Decrease brush size

Algorithms used:
  - MediaPipe 21-point Hand Landmark Detection
  - Euclidean distance-based fingertip tracking
  - Normalised-to-pixel coordinate projection
  - Temporal stroke tracking across frames
  - Frame differencing + overlay blending (OpenCV)
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ---------------------------------------------------------------------------
# MediaPipe Tasks model — auto-downloaded on first run
# ---------------------------------------------------------------------------

_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "hand_landmarker.task")

def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print(f"[Setup] Downloading hand_landmarker.task …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[Setup] Saved to {_MODEL_PATH}")


# Hand connections for skeleton drawing (same topology as the old API)
_HAND_CONNECTIONS = [
    (c.start, c.end)
    for c in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
]


# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe convention)
# ---------------------------------------------------------------------------
WRIST          = 0
THUMB_CMC      = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP      = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP     = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP       = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP      = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP,  INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]


# ---------------------------------------------------------------------------
# Colour palette  (BGR)
# ---------------------------------------------------------------------------
PALETTE = [
    ("Red",     (0,   0,   255)),
    ("Orange",  (0,   140, 255)),
    ("Yellow",  (0,   220, 220)),
    ("Green",   (0,   200,  50)),
    ("Cyan",    (200, 200,   0)),
    ("Blue",    (255,  80,   0)),
    ("Magenta", (220,   0, 220)),
    ("White",   (255, 255, 255)),
]

PALETTE_Y        = 35          # Centre y of palette row
PALETTE_X_START  = 25          # Left edge of first swatch
PALETTE_SPACING  = 55          # Distance between swatch centres
SWATCH_RADIUS    = 20


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def euclidean(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def landmark_px(lm, idx: int, w: int, h: int) -> tuple:
    """
    Project a normalised MediaPipe landmark to pixel coordinates.
    Implements: x_px = lm.x * W,  y_px = lm.y * H
    """
    return int(lm[idx].x * w), int(lm[idx].y * h)


def get_finger_states(lm, w: int, h: int) -> list:
    """
    Return a list of 5 booleans [thumb, index, middle, ring, pinky].
    A finger is considered 'up' when its tip is clearly above its PIP joint
    (lower y value in image coordinates).

    Thumb uses horizontal displacement relative to the IP joint instead
    of vertical, which is more robust for mirrored video.
    """
    states = []

    # Thumb: tip x < IP x  →  extended (mirrored frame, right hand)
    tip_x  = lm[THUMB_TIP].x
    pip_x  = lm[THUMB_IP].x
    states.append(tip_x < pip_x)

    # Four fingers: tip y < PIP y  →  finger pointing up
    for tip_id, pip_id in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        states.append(lm[tip_id].y < lm[pip_id].y)

    return states   # [thumb, index, middle, ring, pinky]


def classify_gesture(fingers: list) -> str:
    """
    Map finger state vector to a gesture label.

      DRAWING : only index up
      CURSOR  : index + middle up (any other fingers may be up too)
      CLEAR   : fist – no fingers up at all
      IDLE    : everything else
    """
    _, index, middle, ring, pinky = fingers

    if not index and not middle and not ring and not pinky:
        return "CLEAR"
    if index and not middle:
        return "DRAWING"
    if index and middle:
        return "CURSOR"
    return "IDLE"


# ---------------------------------------------------------------------------
# Smooth fingertip position with an exponential moving average
# ---------------------------------------------------------------------------

class SmoothPosition:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha          # smoothing factor  (0 = max smooth, 1 = raw)
        self._x: float | None = None
        self._y: float | None = None

    def update(self, x: int, y: int) -> tuple:
        if self._x is None:
            self._x, self._y = float(x), float(y)
        else:
            self._x = self.alpha * x + (1 - self.alpha) * self._x
            self._y = self.alpha * y + (1 - self.alpha) * self._y
        return int(self._x), int(self._y)

    def reset(self):
        self._x = self._y = None


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class AirWriting:

    def __init__(self, cam_id: int = 0, cam_w: int = 1280, cam_h: int = 720):
        # MediaPipe hands (Tasks API — mediapipe >= 0.10)
        _ensure_model()
        _base_opts  = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        _hand_opts  = mp_vision.HandLandmarkerOptions(
            base_options                 = _base_opts,
            running_mode                 = mp_vision.RunningMode.VIDEO,
            num_hands                    = 1,
            min_hand_detection_confidence= 0.75,
            min_hand_presence_confidence = 0.75,
            min_tracking_confidence      = 0.70,
        )
        self.hands = mp_vision.HandLandmarker.create_from_options(_hand_opts)

        # Camera
        self.cam_id = cam_id
        self.cam_w  = cam_w
        self.cam_h  = cam_h
        self.cap    = None

        # Canvas  (created once camera size is known)
        self.canvas: np.ndarray | None = None

        # Drawing state
        self.brush_color     = PALETTE[5][1]   # Blue default
        self.brush_thickness = 6
        self.color_idx       = 5
        self.prev_pt         = None             # last drawn point
        self.smoother        = SmoothPosition(alpha=0.55)

        # Temporal stroke buffer: list of stroke segments for undo (optional)
        self.strokes: list[list[tuple]] = []
        self.current_stroke: list[tuple] = []

        # Gesture debounce
        self._last_gesture  = "IDLE"
        self._gesture_start = 0.0
        self.CLEAR_HOLD_SEC = 0.8   # hold fist this long to actually clear

        # FPS tracking
        self._fps_time  = time.time()
        self._fps_count = 0
        self._fps       = 0.0

    # ------------------------------------------------------------------
    # Hand skeleton drawing (replaces mp_draw.draw_landmarks)
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_hand_landmarks(frame: np.ndarray, lm: list, w: int, h: int):
        pts = [(int(lk.x * w), int(lk.y * h)) for lk in lm]
        for start, end in _HAND_CONNECTIONS:
            cv2.line(frame, pts[start], pts[end], (80, 200, 80), 1, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)
            cv2.circle(frame, pt, 4, (255, 255, 255), 1)

    # ------------------------------------------------------------------
    # Canvas helpers
    # ------------------------------------------------------------------

    def _init_canvas(self, h: int, w: int):
        self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

    def clear_canvas(self):
        if self.canvas is not None:
            self.canvas[:] = 0
        self.prev_pt        = None
        self.strokes.clear()
        self.current_stroke.clear()
        self.smoother.reset()

    def save_canvas(self):
        ts   = time.strftime("%Y%m%d_%H%M%S")
        name = f"air_writing_{ts}.png"
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        cv2.imwrite(path, self.canvas)
        print(f"[Saved] {path}")
        return path

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _draw_stroke(self, pt: tuple):
        """Append point to canvas; uses temporal tracking (prev_pt) for continuity."""
        if self.prev_pt is None:
            # Start a new stroke segment
            cv2.circle(self.canvas, pt, self.brush_thickness // 2,
                       self.brush_color, -1)
            self.current_stroke = [pt]
        else:
            # Draw a thick anti-aliased line between consecutive fingertip positions
            cv2.line(self.canvas, self.prev_pt, pt,
                     self.brush_color, self.brush_thickness, cv2.LINE_AA)
            self.current_stroke.append(pt)

        self.prev_pt = pt

    def _end_stroke(self):
        if self.current_stroke:
            self.strokes.append(self.current_stroke)
            self.current_stroke = []
        self.prev_pt = None
        self.smoother.reset()

    # ------------------------------------------------------------------
    # Colour palette hit-test
    # ------------------------------------------------------------------

    def _check_palette_selection(self, cx: int, cy: int):
        """Select a colour when the cursor hovers over the palette row."""
        for i, (_, clr) in enumerate(PALETTE):
            sx = PALETTE_X_START + i * PALETTE_SPACING
            if euclidean((cx, cy), (sx, PALETTE_Y)) < SWATCH_RADIUS + 8:
                self.color_idx   = i
                self.brush_color = clr
                break

    # ------------------------------------------------------------------
    # UI rendering
    # ------------------------------------------------------------------

    def _draw_palette(self, frame: np.ndarray):
        for i, (name, clr) in enumerate(PALETTE):
            sx = PALETTE_X_START + i * PALETTE_SPACING
            cv2.circle(frame, (sx, PALETTE_Y), SWATCH_RADIUS, clr, -1)
            # Border: thick white for selected, thin grey for others
            border_clr = (255, 255, 255) if i == self.color_idx else (100, 100, 100)
            border_w   = 3              if i == self.color_idx else 1
            cv2.circle(frame, (sx, PALETTE_Y), SWATCH_RADIUS, border_clr, border_w)

    def _draw_hud(self, frame: np.ndarray, gesture: str, fingers: list,
                  clear_progress: float):
        h, w = frame.shape[:2]
        pad = 10

        # --- Gesture badge ---
        badge_colors = {
            "DRAWING": (0,   200,  50),
            "CURSOR":  (0,   165, 255),
            "CLEAR":   (0,    0,  200),
            "IDLE":    (120, 120, 120),
        }
        bg_clr = badge_colors.get(gesture, (120, 120, 120))
        label  = f"  {gesture}  "
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (w - tw - 20, pad), (w - pad, pad + th + 12), bg_clr, -1)
        cv2.putText(frame, label, (w - tw - 14, pad + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # --- Finger indicators ---
        names  = ["T", "I", "M", "R", "P"]
        icon_y = pad + th + 30
        for i, (nm, up) in enumerate(zip(names, fingers)):
            clr = (0, 220, 60) if up else (60, 60, 60)
            cv2.putText(frame, nm, (w - 160 + i * 28, icon_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, clr, 2, cv2.LINE_AA)

        # --- Clear-hold progress bar ---
        if gesture == "CLEAR" and clear_progress > 0:
            bar_w = 200
            bar_h = 14
            bx, by = w // 2 - bar_w // 2, h - 60
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
            fill = int(bar_w * min(clear_progress, 1.0))
            cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), (0, 0, 220), -1)
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (180, 180, 180), 1)
            cv2.putText(frame, "Hold to clear...", (bx, by - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # --- Brush size indicator ---
        bx2, by2 = pad, h - 80
        cv2.putText(frame, f"Brush: {self.brush_thickness}px", (bx2, by2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.circle(frame, (bx2 + 110, by2 - 5),
                   self.brush_thickness // 2 + 1, self.brush_color, -1)

        # --- FPS ---
        cv2.putText(frame, f"FPS: {self._fps:.1f}", (pad, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

        # --- Quick help ---
        tips = [
            "Index UP  = Draw",
            "Index+Mid = Cursor (select colour)",
            "Fist HOLD = Clear",
            "S=Save  C=Clear  Q=Quit",
            "+/-  = Brush size",
        ]
        for i, t in enumerate(tips):
            cv2.putText(frame, t, (pad, h - 35 + i * 16 - (len(tips)-1)*16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Frame blending (OpenCV overlay)
    # ------------------------------------------------------------------

    @staticmethod
    def blend_canvas(frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
        """
        Frame differencing + overlay blend:
          1. Build a binary mask of painted pixels (canvas != black)
          2. Zero-out those pixels in the live frame (bitwise_and with inverted mask)
          3. Add the canvas layer on top  →  opaque drawing over the camera feed
        """
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        mask_inv  = cv2.bitwise_not(mask)

        frame_bg  = cv2.bitwise_and(frame,  frame,  mask=mask_inv)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        return cv2.add(frame_bg, canvas_fg)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        self.cap = cv2.VideoCapture(self.cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)

        if not self.cap.isOpened():
            print("[ERROR] Cannot open webcam. Check camera ID or permissions.")
            return

        ret, frame = self.cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            self.cap.release()
            return

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        self._init_canvas(h, w)

        print("=" * 50)
        print("  Air Writing — OpenCV & MediaPipe")
        print("=" * 50)
        print("  Index UP        → Draw")
        print("  Index + Mid UP  → Cursor / pick colour")
        print("  Fist (hold)     → Clear canvas")
        print("  S → Save  |  C → Clear  |  Q → Quit")
        print("  + / -  → Brush size")
        print("=" * 50)

        cv2.namedWindow("Air Writing", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARNING] Frame grab failed. Retrying …")
                continue

            # Mirror so movement feels natural
            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]

            # Resize canvas if camera resolution changed mid-session
            if self.canvas.shape[:2] != (fh, fw):
                self._init_canvas(fh, fw)

            # ── MediaPipe inference ─────────────────────────────────────
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(time.time() * 1000)
            results   = self.hands.detect_for_video(mp_image, timestamp)

            fingers  = [False] * 5
            gesture  = "IDLE"
            fingertip_px = None

            if results.hand_landmarks:
                lm = results.hand_landmarks[0]

                # Draw skeleton on live frame
                self._draw_hand_landmarks(frame, lm, fw, fh)

                # ── Fingertip tracking ──────────────────────────────────
                fingers = get_finger_states(lm, fw, fh)
                gesture = classify_gesture(fingers)

                # Project index fingertip from normalised → pixel space
                raw_x, raw_y = landmark_px(lm, INDEX_TIP, fw, fh)
                # Smooth position with exponential moving average
                cx, cy       = self.smoother.update(raw_x, raw_y)
                fingertip_px = (cx, cy)

                # ── Gesture actions ─────────────────────────────────────

                if gesture == "DRAWING":
                    self._draw_stroke((cx, cy))

                elif gesture == "CURSOR":
                    self._end_stroke()
                    # Palette hover-select
                    if cy < PALETTE_Y + SWATCH_RADIUS + 15:
                        self._check_palette_selection(cx, cy)

                elif gesture == "CLEAR":
                    self._end_stroke()
                    # Require the user to hold the fist for CLEAR_HOLD_SEC
                    now = time.time()
                    if self._last_gesture != "CLEAR":
                        self._gesture_start = now
                    hold_dur = now - self._gesture_start
                    if hold_dur >= self.CLEAR_HOLD_SEC:
                        self.clear_canvas()
                        self._gesture_start = now  # reset so it doesn't clear instantly again

                else:  # IDLE
                    self._end_stroke()

                self._last_gesture = gesture

                # ── Cursor visualisation ────────────────────────────────
                cursor_clr = self.brush_color
                if gesture == "DRAWING":
                    cv2.circle(frame, fingertip_px,
                               self.brush_thickness // 2 + 4, cursor_clr, 2)
                    cv2.circle(frame, fingertip_px,
                               self.brush_thickness // 2,     cursor_clr, -1)
                elif gesture == "CURSOR":
                    cv2.circle(frame, fingertip_px, 12, cursor_clr, 2)
                    cv2.drawMarker(frame, fingertip_px, cursor_clr,
                                   cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)

            else:
                # No hand detected — end any open stroke
                self._end_stroke()
                self._last_gesture = "IDLE"

            # ── Compute clear-progress for HUD bar ──────────────────────
            clear_progress = 0.0
            if gesture == "CLEAR":
                clear_progress = (time.time() - self._gesture_start) / self.CLEAR_HOLD_SEC

            # ── Overlay blending ─────────────────────────────────────────
            output = self.blend_canvas(frame, self.canvas)

            # ── Draw palette on the blended output ───────────────────────
            self._draw_palette(output)

            # ── HUD overlay ──────────────────────────────────────────────
            self._draw_hud(output, gesture, fingers, clear_progress)

            # ── FPS calculation ──────────────────────────────────────────
            self._fps_count += 1
            now = time.time()
            if now - self._fps_time >= 1.0:
                self._fps       = self._fps_count / (now - self._fps_time)
                self._fps_count = 0
                self._fps_time  = now

            # ── Display ──────────────────────────────────────────────────
            cv2.imshow("Air Writing", output)

            # ── Keyboard controls ─────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):          # Q or ESC → quit
                break
            elif key == ord('s'):
                self.save_canvas()
            elif key == ord('c'):
                self.clear_canvas()
                print("[Canvas cleared]")
            elif key in (ord('+'), ord('=')):
                self.brush_thickness = min(30, self.brush_thickness + 1)
            elif key == ord('-'):
                self.brush_thickness = max(2,  self.brush_thickness - 1)

        # ── Cleanup ───────────────────────────────────────────────────────
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("[Air Writing] Session ended.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Air Writing — OpenCV & MediaPipe")
    parser.add_argument("--cam",    type=int, default=0,    help="Camera device ID (default 0)")
    parser.add_argument("--width",  type=int, default=1280, help="Camera capture width")
    parser.add_argument("--height", type=int, default=720,  help="Camera capture height")
    args = parser.parse_args()

    app = AirWriting(cam_id=args.cam, cam_w=args.width, cam_h=args.height)
    app.run()
