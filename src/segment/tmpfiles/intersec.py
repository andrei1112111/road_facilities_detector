import numpy as np
import cv2

def check_intersection(mask, bbox):
    """
    Проверяет, пересекается ли BBox с маской.
    
    :param mask: numpy array (черно-белая маска, где 1/255 - разметка, 0 - фон)
    :param bbox: tuple (x_min, y_min, x_max, y_max)
    :return: True (есть пересечение), False (нет пересечения)
    """
    x_min, y_min, x_max, y_max = bbox

    # Обрезаем область bbox из маски
    bbox_mask = mask[y_min:y_max, x_min:x_max]

    # Проверяем, есть ли внутри bbox хотя бы один ненулевой пиксель
    return np.any(bbox_mask > 0)  # True, если есть пересечение

# === Тестовый пример ===
# Генерируем маску с размеченной областью
mask = np.zeros((500, 500), dtype=np.uint8)
cv2.rectangle(mask, (200, 200), (300, 300), 255, -1)  # Белый квадрат (разметка)

# Bounding Box
bbox = (250, 250, 350, 350)  # Часть bbox заходит в разметку

# Проверяем пересечение
if check_intersection(mask, bbox):
    print("✅ BBox пересекается с разметкой!")
else:
    print("❌ BBox не пересекается с разметкой!")
