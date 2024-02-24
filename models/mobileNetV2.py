from torchvision import models
import torch
from models.model import Model


class MobileNetV2(Model):
    model = None
    @staticmethod
    def get_instance(*args):
        if not MobileNetV2.model:
            MobileNetV2.model = MobileNetV2(*args)
        return MobileNetV2.model

    def __init__(self, *args):
        model = models.mobilenet_v2(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.to(*args)
        #summary(model,(3,224,224))
        model.eval()
        super().__init__(model,"mobileNetV2",args)

    
    def get_embedding(self, image):
        with torch.no_grad():
            return self.model(image).cpu()