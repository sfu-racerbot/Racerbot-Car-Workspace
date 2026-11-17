#!/usr/bin/env python3
"""
calibrate_intrinsics.py
Calibrates camera intrinsics using a printed checkerboard.
Keys:
  SPACE - Capture valid checkerboard frame
  c     - Compute calibration and save to camera_params.yaml
  q     - Quit
"""

import cv2
import numpy as np
import yaml
import sys

# Checkerboard configuration (inner corners: count rows/cols of internal vertices)
CHECKERBOARD_DIMS = (9, 6)   # e.g., a 10x7 square board has 9x6 inner corners
SQUARE_SIZE_METERS = 0.025   # Length of each checkerboard square side in meters (25mm)
CAMERA_INDEX = 0
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
OUTPUT_YAML = "camera_params.yaml"

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}")
        sys.exit(1)

    # 3D points in real world space for the checkerboard corners
    objp = np.zeros((CHECKERBOARD_DIMS[0] * CHECKERBOARD_DIMS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_DIMS[0], 0:CHECKERBOARD_DIMS[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_METERS

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    captured_count = 0

    print("--- INTRINSICS CALIBRATION ---")
    print("Point the camera at the checkerboard from different angles and distances.")
    print("Press SPACE to save a frame once corners are detected.")
    print("Press 'c' to compute calibration (minimum 15-20 captures recommended).")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD_DIMS,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        display_frame = frame.copy()
        if found:
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD_DIMS, corners, found)
            cv2.putText(display_frame, "Checkerboard Found! Press SPACE to capture", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Checkerboard Not Found", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display_frame, f"Saved frames: {captured_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Calibration Capture", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and found:
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(refined_corners)
            captured_count += 1
            print(f"Captured frame {captured_count}")

        elif key == ord('c'):
            if captured_count < 10:
                print("Capture at least 10-15 frames from varied angles before calibrating.")
                continue

            print("Calibrating camera... please wait...")
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )

            # Compute reprojection error
            total_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
            mean_error = total_error / len(objpoints)

            print(f"Calibration finished!")
            print(f"Mean Reprojection Error: {mean_error:.4f} pixels (target: < 0.5 px)")

            # Save to YAML
            calib_data = {
                "image_width": TARGET_WIDTH,
                "image_height": TARGET_HEIGHT,
                "camera_matrix": K.tolist(),
                "distortion_coefficients": dist.ravel().tolist(),
                "reprojection_error": float(mean_error)
            }
            with open(OUTPUT_YAML, 'w') as f:
                yaml.dump(calib_data, f, default_flow_style=False)
            print(f"Saved calibration parameters to {OUTPUT_YAML}")
            break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()