import cv2

# =========================================
# SOURCE VIDEO SETTINGS
# =========================================

SOURCE_FILE = "video.mp4"

# =========================================
# VIDEO INITIALIZATION
# =========================================

video_stream = cv2.VideoCapture(SOURCE_FILE)

success, current_frame = video_stream.read()

if not success:
    print("Не удалось получить кадр из видео")
    exit()

# =========================================
# FRAME PARAMETERS
# =========================================

frame_h, frame_w = current_frame.shape[:2]

print("=" * 32)
print(f"Ширина кадра : {frame_w}")
print(f"Высота кадра : {frame_h}")
print("=" * 32)

# =========================================
# SPLITTING STEREO IMAGE
# =========================================

center_x = frame_w // 2

camera_left = current_frame[:, 0:center_x]
camera_right = current_frame[:, center_x:frame_w]

# =========================================
# WINDOW DISPLAY
# =========================================

cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
cv2.namedWindow("Left Part", cv2.WINDOW_NORMAL)
cv2.namedWindow("Right Part", cv2.WINDOW_NORMAL)

cv2.imshow("Original Video", current_frame)
cv2.imshow("Left Part", camera_left)
cv2.imshow("Right Part", camera_right)

cv2.waitKey(0)

# =========================================
# CLEANUP
# =========================================

video_stream.release()
cv2.destroyAllWindows()