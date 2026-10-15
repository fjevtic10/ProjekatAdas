# undistort_test_images.py
# Step 2a: apply the camera calibration to a road image
# Run from the Zadatak folder:  python undistort_test_images.py

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

image_name = 'test1.jpg'   # change this to try other test images

# Load calibration (.npz)
calib = np.load('camera_cal/calibration_data.npz')
mtx, dist = calib['mtx'], calib['dist']

img = cv2.imread('test_images/' + image_name)
if img is None:
    raise FileNotFoundError('Could not read test_images/' + image_name)

undistortedImg = cv2.undistort(img, mtx, dist, None, mtx)

os.makedirs('examples', exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image', fontsize=16)
ax2.imshow(cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2RGB))
ax2.set_title('Undistorted Image', fontsize=16)
ax1.axis('off')
ax2.axis('off')
plt.tight_layout()
plt.savefig('examples/undistort_road.png', dpi=100)
plt.close()
print('Saved examples/undistort_road.png')