# perspective.py
# Step 3: perspective transform ("birds-eye view")
# Run from the Zadatak folder:  python perspective.py

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

image_name = 'straight_lines1.jpg'   # also try straight_lines2.jpg

# Load calibration (.npz)
calib = np.load('camera_cal/calibration_data.npz')
mtx, dist = calib['mtx'], calib['dist']

img = cv2.imread('test_images/' + image_name)
if img is None:
    raise FileNotFoundError('Could not read test_images/' + image_name)
print('Image shape:', img.shape)   # expected (720, 1280, 3)

undistortedImg = cv2.undistort(img, mtx, dist, None, mtx)

# Source and destination points for 1280x720 images
# Order: top-left, top-right, bottom-right, bottom-left
SRC = np.float32([[585, 455], [700, 455], [1130, 720], [200, 720]])
DST = np.float32([[320, 0], [960, 0], [960, 720], [320, 720]])

# Perspective transform matrix and its inverse (Minv is used later to unwarp)
M = cv2.getPerspectiveTransform(SRC, DST)
Minv = cv2.getPerspectiveTransform(DST, SRC)

# Warp the undistorted image to the birds-eye view
h, w = undistortedImg.shape[:2]
warped = cv2.warpPerspective(undistortedImg, M, (w, h), flags=cv2.INTER_LINEAR)

# Draw the points to check the transform visually
undistortedMarked = undistortedImg.copy()
cv2.polylines(undistortedMarked, [SRC.astype(np.int32)], True, (0, 0, 255), 3)
warpedMarked = warped.copy()
cv2.polylines(warpedMarked, [DST.astype(np.int32)], True, (0, 0, 255), 3)

os.makedirs('examples', exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.imshow(cv2.cvtColor(undistortedMarked, cv2.COLOR_BGR2RGB))
ax1.set_title('Undistorted with source points', fontsize=14)
ax2.imshow(cv2.cvtColor(warpedMarked, cv2.COLOR_BGR2RGB))
ax2.set_title('Warped with destination points', fontsize=14)
ax1.axis('off')
ax2.axis('off')
plt.tight_layout()
plt.savefig('examples/warped_straight_lines.jpg', dpi=100)
plt.close()
print('Saved examples/warped_straight_lines.jpg')