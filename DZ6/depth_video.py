import cv2
import numpy as np

# =========================================================
# FILES
# =========================================================

INPUT_VIDEO = "video.mp4"
CALIBRATION_FILE = "depth_video.npz"

# =========================================================
# LOAD STEREO PARAMETERS
# =========================================================

calib = np.load(CALIBRATION_FILE)

left_matrix = calib["K1"]
left_dist = calib["D1"]

right_matrix = calib["K2"]
right_dist = calib["D2"]

rotation = calib["R"]
translation = calib["T"]

# =========================================================
# VIDEO SOURCE
# =========================================================

stream = cv2.VideoCapture(INPUT_VIDEO)

status, first_frame = stream.read()

if not status:
    print("Не удалось открыть видеофайл")
    exit()

# =========================================================
# FRAME SPLITTING
# =========================================================

frame_h, frame_w = first_frame.shape[:2]

middle = frame_w // 2

sample_left = first_frame[:, :middle]
sample_right = first_frame[:, middle:]

img_h, img_w = sample_left.shape[:2]

# =========================================================
# STEREO RECTIFICATION
# =========================================================

rect_l, rect_r, proj_l, proj_r, depth_matrix, crop1, crop2 = cv2.stereoRectify(
    left_matrix,
    left_dist,
    right_matrix,
    right_dist,
    (img_w, img_h),
    rotation,
    translation
)

# =========================================================
# UNDISTORTION MAPS
# =========================================================

left_map_x, left_map_y = cv2.initUndistortRectifyMap(
    left_matrix,
    left_dist,
    rect_l,
    proj_l,
    (img_w, img_h),
    cv2.CV_32FC1
)

right_map_x, right_map_y = cv2.initUndistortRectifyMap(
    right_matrix,
    right_dist,
    rect_r,
    proj_r,
    (img_w, img_h),
    cv2.CV_32FC1
)

# =========================================================
# DISPARITY ESTIMATOR
# =========================================================

depth_estimator = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 8,
    blockSize=5,
    P1=8 * 3 * (5 ** 2),
    P2=32 * 3 * (5 ** 2),
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# =========================================================
# MAIN PROCESSING LOOP
# =========================================================

while True:

    success, stereo_frame = stream.read()

    if not success:
        break

    # =====================================================
    # SPLIT STEREO FRAME
    # =====================================================

    left_raw = stereo_frame[:, :middle]
    right_raw = stereo_frame[:, middle:]

    # =====================================================
    # IMAGE RECTIFICATION
    # =====================================================

    fixed_left = cv2.remap(
        left_raw,
        left_map_x,
        left_map_y,
        cv2.INTER_LINEAR
    )

    fixed_right = cv2.remap(
        right_raw,
        right_map_x,
        right_map_y,
        cv2.INTER_LINEAR
    )

    # =====================================================
    # CONVERT TO GRAYSCALE
    # =====================================================

    gray_l = cv2.cvtColor(
        fixed_left,
        cv2.COLOR_BGR2GRAY
    )

    gray_r = cv2.cvtColor(
        fixed_right,
        cv2.COLOR_BGR2GRAY
    )

    # =====================================================
    # CALCULATE DISPARITY
    # =====================================================

    disparity_map = depth_estimator.compute(
        gray_l,
        gray_r
    ).astype(np.float32)

    disparity_map /= 16.0

    # =====================================================
    # NORMALIZATION
    # =====================================================

    disparity_norm = cv2.normalize(
        disparity_map,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    disparity_norm = disparity_norm.astype(np.uint8)

    # =====================================================
    # APPLY COLOR VISUALIZATION
    # =====================================================

    depth_colored = cv2.applyColorMap(
        disparity_norm,
        cv2.COLORMAP_TURBO
    )

    # =====================================================
    # DRAW HELPER LINES
    # =====================================================

    for y in range(0, img_h, 40):
        cv2.line(fixed_left, (0, y), (img_w, y), (0, 255, 0), 1)
        cv2.line(fixed_right, (0, y), (img_w, y), (0, 255, 0), 1)

    # =====================================================
    # DISPLAY WINDOWS
    # =====================================================

    cv2.imshow("Rectified Left Camera", fixed_left)
    cv2.imshow("Rectified Right Camera", fixed_right)
    cv2.imshow("Depth / Disparity", depth_colored)

    pressed = cv2.waitKey(1)

    if pressed == 27:
        break

# =========================================================
# RELEASE RESOURCES
# =========================================================

stream.release()
cv2.destroyAllWindows()