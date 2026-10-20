# measuring_curvature.py
# Step 5: radius of curvature and vehicle position relative to lane center
# Run from the Zadatak folder:  python measuring_curvature.py

import numpy as np
import cv2

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

# ---- Conversions from pixels to meters ----
ym_per_pix = 30 / 720    # meters per pixel in y dimension
xm_per_pix = 3.7 / 700   # meters per pixel in x dimension

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

lab = cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2LAB)
yellow = cv2.inRange(lab[:, :, 2], yellow_min, 255)

hls = cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2HLS)
white = cv2.inRange(hls[:, :, 1], white_min, 255)

gray = cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2GRAY)
sobelX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
absX = cv2.convertScaleAbs(sobelX)
ret, gradient = cv2.threshold(absX, sobel_min, 255, cv2.THRESH_BINARY)

color = cv2.bitwise_or(yellow, white)
binary = cv2.bitwise_or(color, gradient)

mask = np.zeros_like(binary)
cv2.fillPoly(mask, ROI, 255)
binaryRoi = cv2.bitwise_and(binary, mask)

M = cv2.getPerspectiveTransform(SRC, DST)
h, w = binaryRoi.shape[:2]
warped = cv2.warpPerspective(binaryRoi, M, (w, h), flags=cv2.INTER_LINEAR)
ret, binary_warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)

# =====================================================================
# 3. Sliding window search + polynomial fit (pixel space)
# =====================================================================
histogram = np.sum(binary_warped[h // 2:, :], axis=0)
midpoint = w // 2
leftx_base = int(np.argmax(histogram[:midpoint]))
rightx_base = int(np.argmax(histogram[midpoint:])) + midpoint

window_height = h // nwindows

nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

leftx_current = leftx_base
rightx_current = rightx_base
left_lane_inds = []
right_lane_inds = []

for window in range(nwindows):
    win_y_low = h - (window + 1) * window_height
    win_y_high = h - window * window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    if len(good_left_inds) > minpix:
        leftx_current = int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = int(np.mean(nonzerox[good_right_inds]))

left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

if len(leftx) < 3 or len(rightx) < 3:
    raise SystemExit('Not enough lane pixels found - check thresholds / ROI / warp points.')

# Pixel-space fits: x = A*y^2 + B*y + C (used for the lane position)
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# =====================================================================
# 4. Fit polynomials in real-world space (meters)
# =====================================================================
left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

# =====================================================================
# 5. Radius of curvature at the bottom of the image
#    R = (1 + (2*A*y + B)^2)^(3/2) / |2*A|
# =====================================================================
y_eval = h * ym_per_pix

left_radius = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1])**2)**1.5) / abs(2 * left_fit_cr[0])
right_radius = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1])**2)**1.5) / abs(2 * right_fit_cr[0])

# =====================================================================
# 6. Vehicle position relative to the lane center
#    (camera assumed to be mounted at the center of the car)
# =====================================================================
y_bottom = h - 1   # last image row, in pixels
left_x_bottom = left_fit[0] * y_bottom**2 + left_fit[1] * y_bottom + left_fit[2]
right_x_bottom = right_fit[0] * y_bottom**2 + right_fit[1] * y_bottom + right_fit[2]

image_center = w / 2
lane_center = (left_x_bottom + right_x_bottom) / 2
offset = (image_center - lane_center) * xm_per_pix

lane_width_px = right_x_bottom - left_x_bottom
lane_width_m = lane_width_px * xm_per_pix

# =====================================================================
# 7. Results
# =====================================================================
print('----------------------------------------')
print('Image:                  ', image_name)
print('Left line radius:       %.1f m' % left_radius)
print('Right line radius:      %.1f m' % right_radius)
print('Mean radius:            %.1f m' % ((left_radius + right_radius) / 2))
print('Left line x (bottom):   %.1f px' % left_x_bottom)
print('Right line x (bottom):  %.1f px' % right_x_bottom)
print('Lane width:             %.0f px = %.2f m' % (lane_width_px, lane_width_m))
if offset > 0:
    print('Vehicle offset:         %.2f m right of center' % abs(offset))
elif offset < 0:
    print('Vehicle offset:         %.2f m left of center' % abs(offset))
else:
    print('Vehicle offset:         0.00 m (centered)')
print('----------------------------------------')