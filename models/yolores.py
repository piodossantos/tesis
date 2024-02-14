from torchvision import models
import torch
from ultralytics import YOLO
from models.model import Model


class YoloRes(Model):
    model = None
    @staticmethod
    def get_instance(*args):
        if not YoloRes.model:
            YoloRes.model = YoloRes(*args)
        return YoloRes.model

    def __init__(self, *args):
        model = models.resnet18(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.to(args[0])
        #summary(model,(3,224,224))
        model.eval()
        self.model = model
        model2 = YOLO(args[1])
        self.model2 = model2
        super().__init__(model,"yolores",args)

    
    def get_embedding(self, image):
        with torch.no_grad():
            output = self.model(image).cpu().flatten()
            output2 = self.model2.embed(image, verbose=False)[0].cpu()
            return torch.cat((output, output2))