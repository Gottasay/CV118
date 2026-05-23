import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------------
# Настройки
# -----------------------------
VIDEO_FILE = "people.avi"
YOLO_MODEL = "yolov8n.pt"

MIN_CONFIDENCE = 0.35
TRACK_LIFETIME = 30

# скорость воспроизведения:
# чем меньше значение waitKey — тем быстрее видео
PLAYBACK_DELAY = 1

# -----------------------------
# Переменные для линии подсчета
# -----------------------------
selected_points = []
line_created = False


def handle_mouse(event, x, y, flags, userdata):
    """
    Выбор двух точек мышью.
    """
    global selected_points, line_created

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(selected_points) < 2:
            selected_points.append((x, y))

        if len(selected_points) == 2:
            line_created = True


def calculate_side(pt, a, b):
    """
    Определение положения точки
    относительно линии.
    """
    px, py = pt
    ax, ay = a
    bx, by = b

    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)


def determine_region(pt, a, b, tolerance=10):
    """
    Возвращает область точки
    относительно линии.
    """
    value = calculate_side(pt, a, b)

    if value > tolerance:
        return 1

    if value < -tolerance:
        return -1

    return 0


def point_close_to_segment(pt, start, end):
    """
    Проверка близости точки
    к конечному отрезку.
    """
    px, py = pt
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return False

    projection = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    return -0.1 <= projection <= 1.1


def bottom_center(box):
    """
    Нижняя центральная точка bbox.
    """
    left, top, right, bottom = box

    center_x = int((left + right) / 2)
    center_y = int(bottom)

    return center_x, center_y


def render_counter(frame, value):
    """
    Отображение счетчика.
    """
    cv2.rectangle(frame, (10, 10), (220, 70), (40, 40, 40), -1)

    cv2.putText(
        frame,
        f"Crossings: {value}",
        (20, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (255, 255, 0),
        2
    )


def choose_line(first_frame):
    """
    Окно выбора линии.
    """
    global selected_points, line_created

    temp_window = first_frame.copy()

    cv2.namedWindow("Line Setup")
    cv2.setMouseCallback("Line Setup", handle_mouse)

    while True:

        canvas = temp_window.copy()

        for pt in selected_points:
            cv2.circle(canvas, pt, 6, (255, 255, 0), -1)

        if len(selected_points) == 2:
            cv2.line(canvas, selected_points[0], selected_points[1], (0, 120, 255), 3)

        cv2.putText(
            canvas,
            "Select 2 points. ENTER - continue | R - reset",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Line Setup", canvas)

        button = cv2.waitKey(1) & 0xFF

        if button == ord('r'):
            selected_points = []
            line_created = False

        elif button == 13 and line_created:
            break

        elif button == 27:
            return False

    cv2.destroyWindow("Line Setup")
    return True


def run_tracking():

    global selected_points

    stream = cv2.VideoCapture(VIDEO_FILE)

    if not stream.isOpened():
        print("Video open error")
        return

    ok, start_frame = stream.read()

    if not ok:
        print("Cannot read first frame")
        return

    if not choose_line(start_frame):
        stream.release()
        cv2.destroyAllWindows()
        return

    stream.set(cv2.CAP_PROP_POS_FRAMES, 0)

    detector = YOLO(YOLO_MODEL)

    multi_tracker = DeepSort(
        max_age=TRACK_LIFETIME,
        n_init=2,
        max_cosine_distance=0.3,
        nms_max_overlap=1.0
    )

    previous_regions = {}
    already_counted = set()

    crossings_total = 0

    while True:

        success, frame = stream.read()

        if not success:
            break

        predictions = detector(frame, verbose=False)[0]

        tracker_input = []

        if predictions.boxes is not None:

            for detected_box in predictions.boxes:

                class_id = int(detected_box.cls[0].item())
                confidence = float(detected_box.conf[0].item())

                # только люди
                if class_id != 0:
                    continue

                if confidence < MIN_CONFIDENCE:
                    continue

                x1, y1, x2, y2 = map(int, detected_box.xyxy[0].tolist())

                width = x2 - x1
                height = y2 - y1

                tracker_input.append(
                    ([x1, y1, width, height], confidence, "human")
                )

        active_tracks = multi_tracker.update_tracks(
            tracker_input,
            frame=frame
        )

        # линия подсчета
        cv2.line(
            frame,
            selected_points[0],
            selected_points[1],
            (0, 120, 255),
            3
        )

        for tracked in active_tracks:

            if not tracked.is_confirmed():
                continue

            identifier = tracked.track_id

            left, top, right, bottom = map(int, tracked.to_ltrb())

            position = bottom_center([left, top, right, bottom])

            region_now = determine_region(
                position,
                selected_points[0],
                selected_points[1],
                tolerance=12
            )

            near_line = point_close_to_segment(
                position,
                selected_points[0],
                selected_points[1]
            )

            if identifier not in previous_regions:

                if region_now != 0:
                    previous_regions[identifier] = region_now

            else:

                region_before = previous_regions[identifier]

                changed_side = (
                    near_line and
                    region_before != 0 and
                    region_now != 0 and
                    region_before != region_now
                )

                if changed_side and identifier not in already_counted:

                    crossings_total += 1
                    already_counted.add(identifier)

                if region_now != 0:
                    previous_regions[identifier] = region_now

            # bbox
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (255, 140, 0),
                2
            )

            # id
            cv2.putText(
                frame,
                f"p #{identifier}",
                (left, top - 12),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # контрольная точка
            cv2.circle(
                frame,
                position,
                5,
                (0, 0, 255),
                -1
            )

        render_counter(frame, crossings_total)

        cv2.imshow("Smart Counter", frame)

        # УМЕНЬШИЛ СКОРОСТЬ ЗАДЕРЖКИ -> видео идет быстрее
        key = cv2.waitKey(PLAYBACK_DELAY) & 0xFF

        if key == 27:
            break

    stream.release()
    cv2.destroyAllWindows()

    print("-" * 40)
    print(f"Total crossings detected: {crossings_total}")
    print("-" * 40)


if __name__ == "__main__":
    run_tracking()