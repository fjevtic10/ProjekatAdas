# pipeline_video.py
# Step 6: lane overlay on a single image and on a video (any resolution)
# Run from the Zadatak folder:  python pipeline_video.py

import numpy as np
import cv2
import os

# ---- Thresholds ----
yellow_min = 155    # LAB b channel
white_min = 200     # HLS L channel
sobel_min = 100     # |Sobel x| after convertScaleAbs

# ---- Reference geometry, defined for 1280x720 frames ----
# (scaled to the real frame size in get_transform_matrices)
ROI = np.array([[(100, 720), (540, 440), (760, 440), (1200, 720)]], dtype=np.int32)
SRC = np.float32([[585, 455], [700, 455], [1130, 720], [200, 720]])
DST = np.float32([[320, 0], [960, 0], [960, 720], [320, 720]])

# ---- Lane search parameters (for 1280x720, scaled in process_frame) ----
nwindows = 9
margin = 100
minpix = 50
min_pixels = 40         # minimum pixels per line for the search-from-prior result

# ---- Sanity check and smoothing ----
min_lane_px = 200       # lane width in the warped image must stay between these
max_lane_px = 900       # values along the whole image (1280x720), else fit is rejected
n_smooth = 5            # number of accepted fits that are averaged
max_bad_frames = 10     # after this many rejected frames, start again with windows

debug = False           # True: print the lane widths of every rejected fit

# ---- Meter conversions (for 1280x720, scaled in process_frame) ----
ym_per_pix = 30 / 720
xm_per_pix = 3.7 / 700

# ---- Tracking state (shared between frames) ----
left_hist = []          # last accepted left fits
right_hist = []         # last accepted right fits
bad_frames = 0          # consecutive rejected frames

# ---- Calibration (computed on 1280x720 chessboard images) ----
calib = np.load('camera_cal/calibration_data.npz')
mtx, dist = calib['mtx'], calib['dist']


def get_transform_matrices(h, w):
    """Scale ROI, SRC and DST to a frame of size (h, w).
    Returns the scaled ROI polygon, M (warp) and Minv (unwarp)."""
    x_factor = w / 1280.0
    y_factor = h / 720.0
    scale = np.float32([x_factor, y_factor])

    roi_scaled = np.int32(ROI * scale)
    src_scaled = SRC * scale
    dst_scaled = DST * scale

    M = cv2.getPerspectiveTransform(src_scaled, dst_scaled)
    Minv = cv2.getPerspectiveTransform(dst_scaled, src_scaled)
    return roi_scaled, M, Minv


def process_frame(img):
    """Process one BGR frame and return the annotated BGR frame."""
    global bad_frames

    # ---------------- Frame size and scale factors ----------------
    h, w = img.shape[:2]
    x_factor = w / 1280.0
    y_factor = h / 720.0

    roi, M, Minv = get_transform_matrices(h, w)

    margin_px = int(margin * x_factor)
    minpix_px = int(minpix * x_factor * y_factor)
    min_pixels_px = int(min_pixels * x_factor * y_factor)
    min_lane = min_lane_px * x_factor
    max_lane = max_lane_px * x_factor
    ym = ym_per_pix / y_factor      # = 30 / h
    xm = xm_per_pix / x_factor      # = 3.7 / (700 * x_factor)

    # Camera matrix scaled to the frame size (distortion coefficients stay the same)
    mtx_scaled = mtx.copy()
    mtx_scaled[0, 0] = mtx[0, 0] * x_factor    # fx
    mtx_scaled[0, 2] = mtx[0, 2] * x_factor    # cx
    mtx_scaled[1, 1] = mtx[1, 1] * y_factor    # fy
    mtx_scaled[1, 2] = mtx[1, 2] * y_factor    # cy

    # ---------------- Undistort ----------------
    undistortedImg = cv2.undistort(img, mtx_scaled, dist, None, mtx_scaled)

    # ---------------- Thresholding + ROI ----------------
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
    cv2.fillPoly(mask, roi, 255)
    binaryRoi = cv2.bitwise_and(binary, mask)

    # ---------------- Perspective warp ----------------
    warped = cv2.warpPerspective(binaryRoi, M, (w, h), flags=cv2.INTER_LINEAR)
    ret, binary_warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # ---------------- Lane pixels: search from prior ----------------
    have_pixels = False
    if len(left_hist) > 0:
        left_prev = np.mean(left_hist, axis=0)
        right_prev = np.mean(right_hist, axis=0)
        left_center = left_prev[0] * nonzeroy**2 + left_prev[1] * nonzeroy + left_prev[2]
        right_center = right_prev[0] * nonzeroy**2 + right_prev[1] * nonzeroy + right_prev[2]
        left_inds = (nonzerox > left_center - margin_px) & (nonzerox < left_center + margin_px)
        right_inds = (nonzerox > right_center - margin_px) & (nonzerox < right_center + margin_px)
        leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
        rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]
        if len(leftx) >= min_pixels_px and len(rightx) >= min_pixels_px:
            have_pixels = True

    # ---------------- Lane pixels: sliding windows ----------------
    # (always used on the first frame, when there is no previous fit)
    if not have_pixels:
        histogram = np.sum(binary_warped[h // 2:, :], axis=0)
        midpoint = w // 2
        leftx_current = int(np.argmax(histogram[:midpoint]))
        rightx_current = int(np.argmax(histogram[midpoint:])) + midpoint
        window_height = h // nwindows
        left_lane_inds = []
        right_lane_inds = []
        for window in range(nwindows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= leftx_current - margin_px) &
                              (nonzerox < leftx_current + margin_px)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= rightx_current - margin_px) &
                               (nonzerox < rightx_current + margin_px)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > minpix_px:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix_px:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    # ---------------- Polynomial fit + sanity check ----------------
    ploty = np.linspace(0, h - 1, h)
    fit_ok = False
    if len(leftx) >= 3 and len(rightx) >= 3:      # 3 points are needed for a 2nd-order fit
        left_fit_new = np.polyfit(lefty, leftx, 2)
        right_fit_new = np.polyfit(righty, rightx, 2)
        left_fitx_new = left_fit_new[0] * ploty**2 + left_fit_new[1] * ploty + left_fit_new[2]
        right_fitx_new = right_fit_new[0] * ploty**2 + right_fit_new[1] * ploty + right_fit_new[2]
        widths = right_fitx_new - left_fitx_new
        width_ok = widths.min() > min_lane and widths.max() < max_lane
        first_fit = len(left_hist) == 0           # no previous fit: accept the first one
        if width_ok or first_fit:
            fit_ok = True
        elif debug:
            print('Rejected fit: lane width %.0f ... %.0f px (allowed %.0f ... %.0f)'
                  % (widths.min(), widths.max(), min_lane, max_lane))

    if fit_ok:
        left_hist.append(left_fit_new)
        right_hist.append(right_fit_new)
        if len(left_hist) > n_smooth:
            left_hist.pop(0)
            right_hist.pop(0)
        bad_frames = 0
    else:
        bad_frames = bad_frames + 1
        if bad_frames > max_bad_frames:
            left_hist.clear()
            right_hist.clear()
            bad_frames = 0

    # Text size and positions scaled with the frame size
    font_scale = 1.2 * x_factor
    text_x = int(40 * x_factor)
    text_y1 = int(60 * y_factor)
    text_y2 = int(110 * y_factor)

    # No lane pixels found and no previous fit: return the undistorted frame
    if len(left_hist) == 0:
        cv2.putText(undistortedImg, 'Lane not detected', (text_x, text_y1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        return undistortedImg

    # Smoothed fits (average of the last accepted fits)
    left_fit = np.mean(left_hist, axis=0)
    right_fit = np.mean(right_hist, axis=0)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # ---------------- Draw the lane on a blank birds-eye image ----------------
    lane_warp = np.zeros((h, w, 3), dtype=np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(lane_warp, np.int32([pts]), (0, 255, 0))

    # Project it back to the camera perspective and blend with the frame
    lane_unwarped = cv2.warpPerspective(lane_warp, Minv, (w, h))
    result = cv2.addWeighted(undistortedImg, 1, lane_unwarped, 0.3, 0)

    # ---------------- Radius of curvature (meters) ----------------
    left_fit_cr = np.polyfit(ploty * ym, left_fitx * xm, 2)
    right_fit_cr = np.polyfit(ploty * ym, right_fitx * xm, 2)
    y_eval = h * ym
    left_radius = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1])**2)**1.5) / abs(2 * left_fit_cr[0])
    right_radius = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1])**2)**1.5) / abs(2 * right_fit_cr[0])
    radius = (left_radius + right_radius) / 2

    # ---------------- Vehicle offset (meters) ----------------
    lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
    offset = (w / 2 - lane_center) * xm

    # ---------------- Text on the image ----------------
    if radius > 10000:
        radius_text = 'Radius of Curvature = 10000+m'    # practically a straight road
    else:
        radius_text = 'Radius of Curvature = %dm' % radius
    if offset >= 0:
        offset_text = 'Vehicle is %.2fm right of center' % abs(offset)
    else:
        offset_text = 'Vehicle is %.2fm left of center' % abs(offset)

    cv2.putText(result, radius_text, (text_x, text_y1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    cv2.putText(result, offset_text, (text_x, text_y2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    return result


# =====================================================================
# Single test image -> examples/example_output.jpg
# =====================================================================
os.makedirs('examples', exist_ok=True)

testImg = cv2.imread('test_images/test1.jpg')
if testImg is None:
    raise FileNotFoundError('Could not read test_images/test1.jpg')
testResult = process_frame(testImg)
cv2.imwrite('examples/example_output.jpg', testResult)
print('Saved examples/example_output.jpg')

# Reset the tracking state before the video
left_hist.clear()
right_hist.clear()
bad_frames = 0

# =====================================================================
# Video -> project_video_output.mp4
# =====================================================================
# Set video_path directly if your file has a different name
video_candidates = ['test_videos/project_video.mp4',
                    'test_videos/project_video01.mp4',
                    'project_video.mp4']
video_path = None
for candidate in video_candidates:
    if os.path.exists(candidate):
        video_path = candidate
        break

if video_path is None:
    print('Video not found, tried:', video_candidates, '- skipping video.')
else:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Video:', video_path, '-', width, 'x', height, ',', fps, 'fps,', total, 'frames')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('project_video_output.mp4', fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(process_frame(frame))
        frame_count = frame_count + 1
        if frame_count % 50 == 0:
            print('Processed', frame_count, 'of', total, 'frames')

    cap.release()
    out.release()
    print('Saved project_video_output.mp4 (', frame_count, 'frames )')