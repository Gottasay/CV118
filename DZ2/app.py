import cv2
import numpy as np
import sys


def sort_corners(points):

    ordered = np.zeros((4, 2), dtype=np.float32)

    s = points.sum(axis=1)
    ordered[0] = points[np.argmin(s)]
    ordered[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]

    return ordered


def detect_screen(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 60, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 10000:
            continue

        peri = cv2.arcLength(cnt, True)

        poly = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(poly) == 4 and area > best_area:

            best = poly
            best_area = area

    return best


def main():

    insert_path = sys.argv[1]
    base_video = sys.argv[2]

    cap_main = cv2.VideoCapture(base_video)
    cap_insert = cv2.VideoCapture(insert_path)

    fps = cap_main.get(cv2.CAP_PROP_FPS)

    w = int(cap_main.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_main.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        "result.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    ret, first = cap_main.read()

    if not ret:
        print("Ошибка чтения первого кадра")
        return

    screen = detect_screen(first)

    if screen is None:
        print("Не удалось найти экран")
        return

    screen_points = sort_corners(
        screen.reshape(4, 2)
    ).astype(np.float32)

    gray_first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    mask = np.zeros(gray_first.shape, np.uint8)
    cv2.fillConvexPoly(mask, screen_points.astype(int), 255)

    orb = cv2.ORB_create(2000)

    kp_ref, des_ref = orb.detectAndCompute(
        gray_first,
        mask
    )

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ret, frame_insert = cap_insert.read()

    ih, iw = frame_insert.shape[:2]

    src_rect = np.array([
        [0, 0],
        [iw, 0],
        [iw, ih],
        [0, ih]
    ], dtype=np.float32)

    base_transform = cv2.getPerspectiveTransform(
        src_rect,
        screen_points
    )

    while True:

        ret, frame = cap_main.read()

        if not ret:
            break

        ret2, frame_insert = cap_insert.read()

        if not ret2:

            cap_insert.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, frame_insert = cap_insert.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp, des = orb.detectAndCompute(gray, None)

        if des is None:
            writer.write(frame)
            continue

        matches = matcher.match(des_ref, des)

        matches = sorted(matches, key=lambda m: m.distance)

        matches = matches[:150]

        src = np.float32([
            kp_ref[m.queryIdx].pt for m in matches
        ]).reshape(-1, 1, 2)

        dst = np.float32([
            kp[m.trainIdx].pt for m in matches
        ]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(
            src,
            dst,
            cv2.RANSAC,
            5
        )

        if H is None:
            writer.write(frame)
            continue

        total_H = H @ base_transform

        warped = cv2.warpPerspective(
            frame_insert,
            total_H,
            (frame.shape[1], frame.shape[0])
        )

        dst_rect = cv2.perspectiveTransform(
            src_rect.reshape(-1, 1, 2),
            total_H
        )

        mask = np.zeros(frame.shape[:2], np.uint8)

        cv2.fillConvexPoly(
            mask,
            dst_rect.astype(int),
            255
        )

        inv = cv2.bitwise_not(mask)

        background = cv2.bitwise_and(frame, frame, mask=inv)

        foreground = cv2.bitwise_and(warped, warped, mask=mask)

        output = cv2.add(background, foreground)

        cv2.imshow("Result", output)

        writer.write(output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_main.release()
    cap_insert.release()
    writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()