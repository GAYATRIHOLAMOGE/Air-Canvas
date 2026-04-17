# Air Writing

Air Writing is a real-time computer vision project that lets you draw in the air using hand gestures captured through a webcam. Built with Python, OpenCV, and MediaPipe, it tracks your hand using 21-point landmark detection and renders strokes on a virtual canvas — raise your index finger to draw, add your middle finger to move without drawing, make a fist to clear the canvas, and hover over the color panel to switch colors. Brush size can be adjusted with keyboard shortcuts, and the canvas can be saved as a PNG at any time.

## Requirements
- Python 3.x
- OpenCV
- MediaPipe

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python air_writing.py
```

## Controls
| Gesture / Key | Action |
|---|---|
| Index finger up | Draw |
| Index + Middle up | Move cursor & Change color |
| Fist | Clear canvas |
| Hover color panel | Change color |
| `S` | Save canvas as PNG |
| `C` | Clear canvas |
| `+` / `-` | Adjust brush size |
| `Q` / `ESC` | Quit |