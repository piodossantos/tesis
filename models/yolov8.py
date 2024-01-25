from ultralytics import YOLO
from models.model import Model
import torch


class YOLOV8(Model):
    model = None

    @staticmethod
    def get_instance(*args):
        if not YOLOV8.model:
            YOLOV8.model = YOLOV8(*args)
        return YOLOV8.model

    def __init__(self, *args):
        model = YOLO(*args)
        self.model = model
    
    def get_embedding(self, image):
        def make_hook(key):
            def hook(model, input, output):
                intermediate_output[key] = output.detach()
            return hook
        intermediate_output = {}
        self.model.model.model._modules['21'].register_forward_hook(make_hook('21'))
        self.model.predict(image)
        print(self.model.embed(image)[0].shape)
        print(intermediate_output['21'].shape)
        return torch.mean(intermediate_output['21'], (2,3)).squeeze()
