from ultralytics import YOLO
from models.model import Model


class YOLOV8(Model):
    model = None

    @staticmethod
    def get_instance(*args):
        if not YOLOV8.model:
            YOLOV8.model = YOLOV8(*args)
        return YOLOV8.model

    def __init__(self, *args):
        model = YOLO(*args)
        super().__init__(model,"yoloV8",args)
    
    def get_embedding(self, image):
        return self.model.embed(image, verbose=False)[0].cpu()