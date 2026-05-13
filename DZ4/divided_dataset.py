import os
import shutil
from random import shuffle

# Основные директории
images_dir = "dataset/images"
labels_dir = "dataset/labels"

# Папки для разделенного датасета
train_images_dir = "dataset/images/train"
val_images_dir = "dataset/images/val"

train_labels_dir = "dataset/labels/train"
val_labels_dir = "dataset/labels/val"

# Процент тренировочной выборки
train_percent = 0.8

# Создание директорий при отсутствии
for folder in [
    train_images_dir,
    val_images_dir,
    train_labels_dir,
    val_labels_dir
]:
    os.makedirs(folder, exist_ok=True)

# Получение списка изображений
allowed_ext = (".jpg", ".jpeg", ".png")

all_images = [
    file_name
    for file_name in os.listdir(images_dir)
    if file_name.lower().endswith(allowed_ext)
]

# Перемешивание
shuffle(all_images)

# Разделение
border = int(len(all_images) * train_percent)

train_set = all_images[:border]
val_set = all_images[border:]


def move_dataset(files, dst_img_dir, dst_lbl_dir):
    for img_file in files:

        filename = os.path.splitext(img_file)[0]
        txt_file = f"{filename}.txt"

        image_source = os.path.join(images_dir, img_file)
        label_source = os.path.join(labels_dir, txt_file)

        image_target = os.path.join(dst_img_dir, img_file)
        label_target = os.path.join(dst_lbl_dir, txt_file)

        # Проверка существования пары image + label
        if not os.path.isfile(image_source):
            continue

        if not os.path.isfile(label_source):
            print(f"[WARNING] Label not found for: {img_file}")
            continue

        shutil.copy(image_source, image_target)
        shutil.copy(label_source, label_target)


# Копирование train
move_dataset(
    train_set,
    train_images_dir,
    train_labels_dir
)

# Копирование val
move_dataset(
    val_set,
    val_labels_dir,
    val_labels_dir
)

print("=" * 40)
print("Dataset split completed")
print(f"Train images: {len(train_set)}")
print(f"Validation images: {len(val_set)}")
print("=" * 40)