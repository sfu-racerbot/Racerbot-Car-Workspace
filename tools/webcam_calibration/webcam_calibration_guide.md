# Webcam Perception: Calibration & Camera Management Guide

This guide walks through configuring camera driver parameters, computing lens intrinsic calibration, and solving the ground-plane homography matrix (IPM) for 2D position estimation on the racing track.

---

## 1. Prerequisites & Equipment Checklist

### Hardware Requirements
- **Mounted Webcam:** Must be rigidly fixed to the chassis (no flex or vibration).
- **Planar Checkerboard Target:** Standard $9 \times 6$ interior corners (a $10 \times 7$ grid of squares) printed and adhered flat to stiff foam board or acrylic (e.g., $25\,\text{mm}$ or $0.025\,\text{m}$ square pitch).
- **Physical Ground Markers:** High-visibility masking/painter's tape.
- **Metric Tape Measure:** Capable of measuring out to at least $5\text{--}10\,\text{m}$.

### Software Dependencies
Install the required system utilities and Python dependencies on your Linux host/onboard computer:
```bash
sudo apt update
sudo apt install -y v4l-utils python3-opencv python3-yaml python3-numpy
```

Ensure the setup script has execution permissions:
```bash
chmod +x set_camera_controls.sh
```

---

## 2. Step 1: Camera Driver Configuration (`set_camera_controls.sh`)

Dynamic auto-exposure, auto-white balance, and autofocus cause dramatic lighting and color shifts that break HSV wall segmentation and feature tracking. These must be locked at the driver level before collecting any calibration images or running perception.

### Identify Camera Device
List connected video interfaces:
```bash
v4l2-ctl --list-devices
```
Locate your webcam index (typically `/dev/video0` or `/dev/video2`).

### Run Driver Lock
```bash
./set_camera_controls.sh /dev/video0
```

### Manual Adjustments (If Needed)
If the image appears too dark or overexposed in your track lighting environment:
```bash
# View all supported controls and valid value ranges
v4l2-ctl -d /dev/video0 -l

# Adjust exposure time (higher = brighter, lower = less motion blur)
v4l2-ctl -d /dev/video0 -c exposure_time_absolute=250

# Adjust sensor gain
v4l2-ctl -d /dev/video0 -c gain=35
```

---

## 3. Step 2: Camera Intrinsic Calibration (`calibrate_intrinsics.py`)

This step solves for focal length $(f_x, f_y)$, optical center $(c_x, c_y)$, and lens radial/tangential distortion coefficients $(k_1, k_2, p_1, p_2)$ to rectify lens distortion.

### Configuration
Verify parameters at the top of `calibrate_intrinsics.py` match your physical board:
```python
CHECKERBOARD_DIMS = (9, 6)   # Inner corners: (columns - 1, rows - 1)
SQUARE_SIZE_METERS = 0.025   # Physical square side length in meters
CAMERA_INDEX = 0             # 0 for /dev/video0
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
```

### Execution
Run the interactive calibration script:
```bash
python3 calibrate_intrinsics.py
```

### Capture Protocol
1. Hold the checkerboard in front of the lens. The display will draw colored corner lines once detected.
2. Press **SPACEBAR** to record a valid capture.
3. Capture **15 to 25 frames** covering:
   - **Range:** Close ($0.5\,\text{m}$), medium ($1.5\,\text{m}$), and far ($2.5\,\text{m}$).
   - **Angles:** Tilt the board forward/backward and left/right ($15^\circ\text{--}30^\circ$).
   - **Viewport Distribution:** Move the board to all 4 corners and edges of the frame to capture optical perimeter distortion.
4. Press **`c`** to execute the optimization solver.
5. **Quality Check:** Review the terminal output for the **Mean Reprojection Error**:
   - Ideal: $< 0.35\,\text{pixels}$
   - Acceptable: $< 0.50\,\text{pixels}$
   - Poor: $> 0.70\,\text{pixels}$ (discard and re-run with clearer captures).
6. The script writes results directly to `camera_params.yaml`.

---

## 4. Step 3: Ground Homography / IPM Calibration (`calibrate_homography.py`)

Inverse Perspective Mapping maps rectified image pixels $(u, v)$ to real-world ground positions $(X_w, Y_w)$ on the flat track surface ($Z_w = 0$).

### Track Setup
1. Position the vehicle on a level, flat surface.
2. Define the vehicle origin: ground point centered directly below the front axle / camera mount.
3. Using a tape measure, lay 4 tape squares at measured metric coordinates $(X, Y)$:
   - **P1:** $1.0\,\text{m}$ forward, $0.5\,\text{m}$ left $\rightarrow (+1.0, +0.5)$
   - **P2:** $1.0\,\text{m}$ forward, $0.5\,\text{m}$ right $\rightarrow (+1.0, -0.5)$
   - **P3:** $3.0\,\text{m}$ forward, $0.5\,\text{m}$ right $\rightarrow (+3.0, -0.5)$
   - **P4:** $3.0\,\text{m}$ forward, $0.5\,\text{m}$ left $\rightarrow (+3.0, +0.5)$

> *Note: If space constraints require different distances, update the `WORLD_POINTS` array inside `calibrate_homography.py` before running.*

### Execution
```bash
python3 calibrate_homography.py
```

### Marking Protocol
1. The script will automatically load `camera_params.yaml` and display a rectified stream.
2. In the OpenCV window, click the center of each taped ground marker in strict counter-clockwise order:
   - Click 1 $\rightarrow$ **P1** ($1\,\text{m}$ ahead, Left)
   - Click 2 $\rightarrow$ **P2** ($1\,\text{m}$ ahead, Right)
   - Click 3 $\rightarrow$ **P3** ($3\,\text{m}$ ahead, Right)
   - Click 4 $\rightarrow$ **P4** ($3\,\text{m}$ ahead, Left)
3. A polygon will enclose the selected region.
   - Press **`r`** if you misclicked to reset points.
   - Press **`s`** to compute the homography and save `homography.yaml`.

---

## 5. Verification & Sanity Check

To verify that the calculated homography provides accurate metric measurements before integrating with full tracking:

1. Place a stationary test target (such as an opponent RC car or traffic cone) at a known distance (e.g., $X = 2.0\,\text{m}, Y = 0.0\,\text{m}$).
2. Run a quick validation test in Python:
```python
import numpy as np
import yaml

# Load IPM mapping
with open("homography.yaml", "r") as f:
    H_inv = np.array(yaml.safe_load(f)["H_image_to_ground"], dtype=np.float32)

# Input bottom-center pixel (u, v) where tires touch the asphalt
u, v = 640.0, 510.0  # Replace with actual observed pixel coordinates
img_pt = np.array([u, v, 1.0], dtype=np.float32).reshape(3, 1)

world_pt = H_inv @ img_pt
scale = world_pt[2, 0]

X_w = world_pt[0, 0] / scale
Y_w = world_pt[1, 0] / scale

print(f"Calculated Metric Position -> Forward: {X_w:.2f} m, Lateral: {Y_w:.2f} m")
```
3. Compare the calculated coordinates with a tape measure or live LiDAR reading. Distance error within $1\text{--}5\,\text{m}$ should remain within $\pm 5\text{--}10\,\text{cm}$.

---

## 6. Failure Modes & Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **High reprojection error ($>0.7\,\text{px}$)** | Blurry captures or bowed checkerboard | Ensure board is completely flat and hold steady when pressing SPACEBAR. |
| **Checkerboard not detected** | Glare, reflections, or too close to frame edge | Adjust room lighting or tilt board slightly to reduce specular reflection. |
| **Calculated IPM distances jump or diverge** | Bounding box clipped or selected point near horizon | Check that $v_{\text{contact}} < y_{\text{horizon}} - \text{margin}$ and bottom boundary is not cut off by screen edge. |
| **Calculated distance is off under motion** | Vehicle suspension pitch during acceleration/braking | Integrate IMU pitch angle to dynamically adjust pitch in rotation matrix $R$. |