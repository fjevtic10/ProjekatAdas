# line_fit.py
# Step 4: lane-line detection (sliding windows) and 2nd-order polynomial fit,
#         plus search_around_poly() for tracking in video
# Run from the Zadatak folder:  python line_fit.py

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

image_name = 'test1.jpg'

# ---- Thresholds ----
yellow_min = 155    # LAB b channel
white_min = 200     # HLS L channel
sobel_min = 100     # |Sobel x| after convertScaleAbs

# ---- Region of interest (polygon, 1280x720 image) ----
ROI = np.array([[(100, 720), (540, 440), (760, 440), (1200, 720)]], dtype=np.int32)

# ---- Perspective transform points (1280x720) ----
SRC = np.float32([[585, 455], [700, 455], [1130, 720], [200, 720]])
DST = np.float32([[320, 0], [960, 0], [960, 720], [320, 720]])

# ---- Sliding window parameters ----
nwindows = 9
margin = 100
minpix = 50


def search_around_poly(binary_warped, left_fit, right_fit, margin=100):
    """Search for lane pixels only within +/- margin of the previous fit.
    Returns the pixels and the new fits (None if too few pixels were found)."""
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_center = left_fit[0] * nonzeroy**2 + left_fit[1] * nonzeroy + left_fit[2]
    right_center = right_fit[0] * nonzeroy**2 + right_fit[1] * nonzeroy + right_fit[2]

    left_inds = (nonzerox > left_center - margin) & (nonzerox < left_center + margin)
    right_inds = (nonzerox > right_center - margin) & (nonzerox < right_center + margin)

    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    if len(leftx) < 3 or len(rightx) < 3:
        return leftx, lefty, rightx, righty, None, None

    left_fit_new = np.polyfit(lefty, leftx, 2)
    right_fit_new = np.polyfit(righty, rightx, 2)
    return leftx, lefty, rightx, righty, left_fit_new, right_fit_new


# =====================================================================
# 1. Load calibration and the test image
# =====================================================================
calib = np.load('camera_cal/calibration_data.npz')
mtx, dist = calib['mtx'], calib['dist']

img = cv2.imread('test_images/' + image_name)
if img is None:
    raise FileNotFoundError('Could not read test_images/' + image_name)

# =====================================================================
# 2. Undistort, threshold, ROI mask, perspective warp
# =====================================================================
undistortedImg = cv2.undistort(img, mtx, dist, None, mtx)

# Yellow: b channel of LAB
lab = cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2LAB)
yellow = cv2.inRange(lab[:, :, 2], yellow_min, 255)

# White: lightness channel of HLS
hls = cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2HLS)
white = cv2.inRange(hls[:, :, 1], white_min, 255)

# Gradient: Sobel x on grayscale
gray = cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2GRAY)
sobelX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
absX = cv2.convertScaleAbs(sobelX)
ret, gradient = cv2.threshold(absX, sobel_min, 255, cv2.THRESH_BINARY)

# Combine: (yellow OR white) OR gradient
color = cv2.bitwise_or(yellow, white)
binary = cv2.bitwise_or(color, gradient)

# Keep only the region of interest
mask = np.zeros_like(binary)
cv2.fillPoly(mask, ROI, 255)
binaryRoi = cv2.bitwise_and(binary, mask)

# Birds-eye view (values 0 or 255 again after interpolation)
M = cv2.getPerspectiveTransform(SRC, DST)
Minv = cv2.getPerspectiveTransform(DST, SRC)
h, w = binaryRoi.shape[:2]
warped = cv2.warpPerspective(binaryRoi, M, (w, h), flags=cv2.INTER_LINEAR)
ret, binary_warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)

# =====================================================================
# 3. Histogram of the bottom half -> base of the left and right lines
# =====================================================================
histogram = np.sum(binary_warped[h // 2:, :], axis=0)
midpoint = w // 2
leftx_base = int(np.argmax(histogram[:midpoint]))
rightx_base = int(np.argmax(histogram[midpoint:])) + midpoint
print('Lane base positions: left x =', leftx_base, ', right x =', rightx_base)

# =====================================================================
# 4. Sliding window search
# =====================================================================
out_img = np.dstack((binary_warped, binary_warped, binary_warped))  # for drawing

window_height = h // nwindows

nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

leftx_current = leftx_base
rightx_current = rightx_base
left_lane_inds = []
right_lane_inds = []

for window in range(nwindows):
    # Window boundaries (start at the bottom and go up)
    win_y_low = h - (window + 1) * window_height
    win_y_high = h - window * window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    # Draw the windows on the visualization image
    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 3)
    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 3)

    # Nonzero pixels inside each window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    # Re-center the next window if enough pixels were found
    if len(good_left_inds) > minpix:
        leftx_current = int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = int(np.mean(nonzerox[good_right_inds]))

left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
print('Sliding windows: left pixels =', len(leftx), ', right pixels =', len(rightx))

if len(leftx) < 3 or len(rightx) < 3:
    raise SystemExit('Not enough lane pixels found - check thresholds / ROI / warp points.')

# =====================================================================
# 5. Fit x = A*y^2 + B*y + C for both lines
# =====================================================================
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
print('Left  fit [A, B, C]:', left_fit)
print('Right fit [A, B, C]:', right_fit)

ploty = np.linspace(0, h - 1, h)
left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

out_img[lefty, leftx] = [255, 0, 0]      # left line pixels: red
out_img[righty, rightx] = [0, 0, 255]    # right line pixels: blue

# =====================================================================
# 6. Search around the previous polynomial (as used for video frames)
#    Here the "previous" fit is the sliding-window fit of this same image.
# =====================================================================
leftx2, lefty2, rightx2, righty2, left_fit2, right_fit2 = search_around_poly(
    binary_warped, left_fit, right_fit, margin)

if left_fit2 is None or right_fit2 is None:
    print('search_around_poly: too few pixels, keeping the previous fit')
    left_fit2, right_fit2 = left_fit, right_fit
print('Prior search: left pixels =', len(leftx2), ', right pixels =', len(rightx2))

left_fitx2 = left_fit2[0] * ploty**2 + left_fit2[1] * ploty + left_fit2[2]
right_fitx2 = right_fit2[0] * ploty**2 + right_fit2[1] * ploty + right_fit2[2]

# Visualization image: pixels found by the prior search + shaded margin region
out_img2 = np.dstack((binary_warped, binary_warped, binary_warped))
out_img2[lefty2, leftx2] = [255, 0, 0]
out_img2[righty2, rightx2] = [0, 0, 255]

window_img = np.zeros_like(out_img2)
# The shaded region is the search area around the PREVIOUS fit (left_fitx, right_fitx)
left_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
left_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
left_pts = np.hstack((left_window1, left_window2))
right_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
right_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
right_pts = np.hstack((right_window1, right_window2))
cv2.fillPoly(window_img, np.int32([left_pts]), (0, 255, 0))
cv2.fillPoly(window_img, np.int32([right_pts]), (0, 255, 0))
result = cv2.addWeighted(out_img2, 1, window_img, 0.3, 0)

# =====================================================================
# 7. Save the visualization
# =====================================================================
os.makedirs('examples', exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.imshow(out_img)
ax1.plot(left_fitx, ploty, color='yellow', linewidth=3)
ax1.plot(right_fitx, ploty, color='yellow', linewidth=3)
ax1.set_title('Sliding windows + fitted polynomials', fontsize=14)

ax2.imshow(result)
ax2.plot(left_fitx2, ploty, color='yellow', linewidth=3)
ax2.plot(right_fitx2, ploty, color='yellow', linewidth=3)
ax2.set_title('Search from prior (green = margin)', fontsize=14)

for ax in (ax1, ax2):
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')
plt.tight_layout()
plt.savefig('examples/color_fit_lines.jpg', dpi=100)
plt.close()
print('Saved examples/color_fit_lines.jpg')