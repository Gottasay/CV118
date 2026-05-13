from ultralytics import YOLO
import cv2

# Загрузка обученной модели
detector = YOLO("runs/detect/train/weights/best.pt")

# Подключение к веб-камере
camera = cv2.VideoCapture(0)

window_name = "Robot Detection"

while camera.isOpened():

    success, image = camera.read()

    if not success:
        print("Не удалось получить кадр")
        break

    # Выполнение детекции
    prediction = detector.predict(source=image, verbose=True)

    # Получение кадра с отрисованными объектами
    rendered_frame = prediction[0].plot()

    # Вывод изображения
    cv2.imshow(window_name, rendered_frame)

    key = cv2.waitKey(1) & 0xFF

    # ESC
    if key == 27:
        break

# Освобождение ресурсов
camera.release()
cv2.destroyAllWindows()