# calibrate.py
# Step 1: camera calibration with chessboard images
# Run from the Zadatak folder:  python calibrate.py

import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

# Number of inner corners of the chessboard
rows = 9
cols = 6
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 3D points of the corners: (0,0,0), (1,0,0), ... (8,5,0)
objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

objectPointsArray = []
imgPointsArray = []

paths = sorted(glob.glob('camera_cal/calibration*.jpg'))
print('Found', len(paths), 'calibration images')

for path in paths:
    img = cv2.imread(path)
    if img is None:
        print('[SKIP] could not read', path)
        continue

    # two images of the set are 1281x721, resize them to 1280x720
    if img.shape[:2] != (720, 1280):
        img = cv2.resize(img, (1280, 720))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objectPointsArray.append(objectPoints)
        imgPointsArray.append(corners)
        print('[OK]  ', path)
    else:
        print('[FAIL]', path, '- corners not found')

if len(objectPointsArray) == 0:
    raise SystemExit('No chessboard corners found in any image.')

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)

print('Used', len(objectPointsArray), 'of', len(paths), 'images')
print('RMS reprojection error:', ret)
print('Camera matrix:\n', mtx)
print('Distortion coefficients:', dist.ravel())

np.savez('camera_cal/calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print('Saved camera_cal/calibration_data.npz')

# Example of distortion correction on a calibration image
img = cv2.imread('camera_cal/calibration1.jpg')
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
plt.savefig('examples/undistort_output.png', dpi=100)
plt.close()
print('Saved examples/undistort_output.png')