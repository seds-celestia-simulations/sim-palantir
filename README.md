# 3D Electromagnetic Wave Simulator

A simple 3D electromagnetic wave simulator using Maxwell's equations and the Yee finite-difference time-domain (FDTD) method. Built with Python, NumPy, Pygame, and OpenGL.

## Run

```bash
pip install -r requirements.txt
python yee_viewer.py
```

Keep `yee_physics.py` and `yee_viewer.py` in the same directory.

## Implementation

- **`yee_physics.py`** implements Maxwell's equations using vectorized NumPy operations, finite-difference curl operators, and a time-stepping loop.
- **`yee_viewer.py`** renders the 3D electric field using Pygame and OpenGL, with coordinate axes and a source marker.

The simulation uses normalized units, periodic boundaries, and an idealized current source.

## Controls

- **Left-click + drag:** Rotate the view
- **Scroll:** Zoom
- **Space:** Pause/resume
- **Escape:** Exit