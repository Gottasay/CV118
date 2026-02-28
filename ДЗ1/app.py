import sys
import cv2
from tkinter import *
from PIL import Image, ImageTk


class VideoApp:
    def __init__(self, root, video_source):
        self.root = root
        self.root.title("Video Application")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Видео источник
        if video_source == "camera":
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(video_source)

        if not self.cap.isOpened():
            print("Ошибка открытия видео источника")
            sys.exit(1)

        # Холст для отображения видео
        self.canvas = Label(root)
        self.canvas.pack()

        # Кнопка выхода
        self.quit_button = Button(root, text="Quit (Q)", command=self.on_close)
        self.quit_button.pack()

        # Список точек
        self.points = []

        # Привязка событий
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.root.bind("<Key>", self.on_key_press)

        self.update_frame()

    def on_mouse_click(self, event):
        # Сохраняем координаты клика
        self.points.append((event.x, event.y))

    def on_key_press(self, event):
        if event.char.lower() == 'c':
            # Сброс точек
            self.points.clear()
        elif event.char.lower() == 'q':
            self.on_close()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Рисуем прямоугольники в местах кликов
            for (x, y) in self.points:
                cv2.rectangle(frame,
                              (x - 20, y - 20),
                              (x + 20, y + 20),
                              (255, 0, 0),
                              2)

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def on_close(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование:")
        print("python app.py camera")
        print("или")
        print("python app.py video.mp4")
        sys.exit(1)

    video_source = sys.argv[1]

    root = Tk()
    app = VideoApp(root, video_source)
    root.mainloop()