#!/bin/bash
# set_camera_controls.sh
# Usage: ./set_camera_controls.sh /dev/video0

DEVICE=${1:-/dev/video0}

echo "Configuring camera device: $DEVICE"

# Ensure v4l2-ctl is installed
if ! command -v v4l2-ctl &> /dev/null; then
    echo "Error: v4l2-ctl could not be found. Run: sudo apt install v4l-utils"
    exit 1
fi

# 1. Disable auto white balance (1 = auto, 0 = manual)
v4l2-ctl -d "$DEVICE" -c white_balance_automatic=0 2>/dev/null || true
# Set a fixed white balance temperature (typical indoor lighting ~4500K)
v4l2-ctl -d "$DEVICE" -c white_balance_temperature=4500 2>/dev/null || true

# 2. Disable auto exposure (V4L2: 1 = Manual Mode, 3 = Aperture Priority / Auto)
v4l2-ctl -d "$DEVICE" -c auto_exposure=1 2>/dev/null || true
# Set manual exposure time (adjust between 100 and 500 depending on lighting)
v4l2-ctl -d "$DEVICE" -c exposure_time_absolute=200 2>/dev/null || true

# 3. Set fixed gain
v4l2-ctl -d "$DEVICE" -c gain=30 2>/dev/null || true

# 4. Disable auto-focus if supported (0 = manual)
v4l2-ctl -d "$DEVICE" -c focus_automatic_continuous=0 2>/dev/null || true
v4l2-ctl -d "$DEVICE" -c focus_absolute=0 2>/dev/null || true

echo "Camera controls locked successfully on $DEVICE."