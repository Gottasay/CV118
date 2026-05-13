import cv2
import os


video_path = "videorobot.mp4"   # путь к видео
output_folder = "raw_frames"    # куда сохраняем кадры
frame_step = 5                  # берём каждый N-й кадр


os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка: не удалось открыть видео")
    exit()

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # сохраняем каждый frame_step кадр
    if frame_count % frame_step == 0:
        filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f" Сохранено {saved_count} кадров в папку '{output_folder}'")