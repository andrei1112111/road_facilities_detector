from ultralytics import YOLO


class PitsAndCracks:
    """detection of potholes and cracks"""

    def __init__(self, class_numbers, model):
        """init model"""
        self.model = YOLO(model)
        self.conf = 0.25
        self.imgsz=640  # на котором обучались


    def predict(self, frame):
        """get frame and return bboxes of potholes and cracks"""
        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)

        # Получаем количество обнаруженных объектов
        num_boxes = len(results[0].boxes)

        # Наложение боксов на кадр
        # annotated_frame = results[0].plot()

        # <0 фигня
        # = 0 нет дороги
        # > 0 хорошая
        return 1 if num_boxes == 0 else -1
