#!/usr/bin/env python3
"""
calibrate_homography.py
Computes the Inverse Perspective Mapping (IPM) homography matrix
by mapping clicked pixel coordinates to known real-world track coordinates.
"""

import cv2
import numpy as np
import yaml
import sys

# Define 4 real-world ground points (X = forward distance in meters, Y = lateral distance in meters, positive left)
# Place tape markers on the ground relative to your car's front axle/camera mount center.
WORLD_POINTS = np.array([
    [1.0,  0.5],   # Point 1: 1.0m forward, 0.5m left
    [1.0, -0.5],   # Point 2: 1.0m forward, 0.5m right
    [3.0, -0.5],   # Point 3: 3.0m forward, 0.5m right
    [3.0,  0.5],   # Point 4: 3.0m forward, 0.5m left
], dtype=np.float32)

OUTPUT_YAML = "homography.yaml"
CAMERA_INDEX = 0
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

clicked_pixels = []

def mouse_callback(event, x, y, flags, param):
    global clicked_pixels
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pixels) < 4:
        clicked_pixels.append([float(x), float(y)])
        print(f"Clicked point {len(clicked_pixels)}/4 at pixel: ({x}, {y})")

def main():
    global clicked_pixels
    
    # Load intrinsics if available to undistort frame
    K, dist = None, None
    try:
        with open("camera_params.yaml", 'r') as f:
            data = yaml.safe_load(f)
            K = np.array(data["camera_matrix"], dtype=np.float32)
            dist = np.array(data["distortion_coefficients"], dtype=np.float32)
            print("Loaded camera_params.yaml for distortion rectification.")
    except FileNotFoundError:
        print("Warning: camera_params.yaml not found. Running on raw distorted frame.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}")
        sys.exit(1)

    print("\n--- HOMOGRAPHY (IPM) CALIBRATION ---")
    print("1. Align the vehicle straight on a flat floor.")
    print("2. Click the 4 ground points in this EXACT order:")
    for idx, pt in enumerate(WORLD_POINTS):
        print(f"   Point {idx+1}: Forward X={pt[0]}m, Lateral Y={pt[1]}m")
    print("Press 'r' to reset points, 's' to compute & save, 'q' to quit.")

    cv2.namedWindow("Homography Calibration")
    cv2.setMouseCallback("Homography Calibration", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if K is not None and dist is not None:
            frame = cv2.undistort(frame, K, dist)

        display_frame = frame.copy()

        # Draw clicked points
        for i, pt in enumerate(clicked_pixels):
            cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
            cv2.putText(display_frame, f"P{i+1}", (int(pt[0]) + 8, int(pt[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw polygon connecting selected points if 4 points selected
        if len(clicked_pixels) == 4:
            pts_poly = np.array(clicked_pixels, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts_poly], isClosed=True, color=(255, 255, 0), thickness=2)
            cv2.putText(display_frame, "4 points captured! Press 's' to save.", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, f"Click point {len(clicked_pixels)+1}/4", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("Homography Calibration", display_frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            clicked_pixels = []
            print("Points reset.")

        elif key == ord('s'):
            if len(clicked_pixels) != 4:
                print("Must click exactly 4 points before saving.")
                continue

            src_img_pts = np.array(clicked_pixels, dtype=np.float32)
            dst_world_pts = WORLD_POINTS

            # H_inv maps directly from Image Pixels -> World Ground Coordinates (X_w, Y_w)
            H_inv, status = cv2.findHomography(src_img_pts, dst_world_pts)

            calib_data = {
                "H_image_to_ground": H_inv.tolist(),
                "world_points_reference": dst_world_pts.tolist(),
                "clicked_image_points": src_img_pts.tolist()
            }

            with open(OUTPUT_YAML, 'w') as f:
                yaml.dump(calib_data, f, default_flow_style=False)

            print(f"Successfully saved homography to {OUTPUT_YAML}!")
            break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()