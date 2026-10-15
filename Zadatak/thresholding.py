# thresholding.py
# Step 2b: color + gradient thresholding with ROI mask to find yellow and white lane lines
# Run from the Zadatak folder: python thresholding.py

import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

# ---- Thresholds ----
yellow_min = 155     # LAB b channel: high values = yellow
white_min = 200      # HLS L channel: high values = bright / white
sobel_min = 100      # |Sobel x| threshold

example_name = 'test1.jpg'   # image used for examples/binary_combo_example.jpg

# Load calibration (.npz)
calib = np.load('camera_cal/calibration_data.npz')
mtx, dist = calib['mtx'], calib['dist']

paths = sorted(glob.glob('test_images/*.jpg'))
if len(paths) == 0:
    raise FileNotFoundError('No .jpg files in test_images/')

os.makedirs('examples', exist_ok=True)

# Figure with every test image next to its binary
gridFig, gridAxes = plt.subplots(len(paths), 2, figsize=(10, 2.8 * len(paths)))
gridAxes = np.atleast_2d(gridAxes)

exampleSaved = False

for i, path in enumerate(paths):
    img = cv2.imread(path)
    undistortedImg = cv2.undistort(img, mtx, dist, None, mtx)
    h, w = undistortedImg.shape[:2]

    # Region of Interest (ROI) mask: keep only bottom portion of the image (road)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    # Define polygon mask for the road region
    roi_vertices = np.array([[(100, h), (w//2 - 50, int(h*0.55)), (w//2 + 50, int(h*0.55)), (w - 50, h)]], dtype=np.int32)
    cv2.fillPoly(roi_mask, roi_vertices, 255)

    # Yellow lines: b channel of LAB
    lab = cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2LAB)
    yellow = cv2.inRange(lab[:, :, 2], yellow_min, 255)

    # White lines: lightness (L) channel of HLS
    hls = cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls[:, :, 1], white_min, 255)

    # Gradient: Sobel in x direction on the grayscale image
    gray = cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2GRAY)
    sobelX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
    absX = cv2.convertScaleAbs(sobelX)
    ret, gradient = cv2.threshold(absX, sobel_min, 255, cv2.THRESH_BINARY)

    # Combine color components and gradient
    color = cv2.bitwise_or(yellow, white)
    combined = cv2.bitwise_or(color, gradient)

    # Apply ROI mask to clean up sky, trees, and environment noise
    binary = cv2.bitwise_and(combined, roi_mask)

    gridAxes[i, 0].imshow(cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2RGB))
    gridAxes[i, 0].set_title(os.path.basename(path), fontsize=10)
    gridAxes[i, 1].imshow(binary, cmap='gray')
    gridAxes[i, 0].axis('off')
    gridAxes[i, 1].axis('off')

    # Example figure: undistorted | stacked (green = gradient, blue = color) | binary
    if os.path.basename(path) == example_name:
        # Mask gradient and color for clean visualization
        gradient_masked = cv2.bitwise_and(gradient, roi_mask)
        color_masked = cv2.bitwise_and(color, roi_mask)
        
        stacked = np.dstack((np.zeros_like(gradient_masked), gradient_masked, color_masked))
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        axes[0].imshow(cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Undistorted', fontsize=14)
        axes[1].imshow(stacked)
        axes[1].set_title('Green: gradient | Blue: color', fontsize=14)
        axes[2].imshow(binary, cmap='gray')
        axes[2].set_title('Combined binary', fontsize=14)
        for ax in axes:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig('examples/binary_combo_example.jpg', dpi=100)
        plt.close(fig)
        print('Saved examples/binary_combo_example.jpg')
        exampleSaved = True

if not exampleSaved:
    print('WARNING: ' + example_name + ' not found in test_images/')

gridFig.tight_layout()
gridFig.savefig('examples/binary_all_test_images.png', dpi=80)
plt.close(gridFig)
print('Saved examples/binary_all_test_images.png')