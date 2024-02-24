from torchvision import models
import torch
from models.model import Model


class Resnet152(Model):
    model = None
    @staticmethod
    def get_instance(*args):
        if not Resnet152.model:
            Resnet152.model = Resnet152(*args)
        return Resnet152.model

    def __init__(self, *args):
        model = models.resnet152(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.to(*args)
        #summary(model,(3,224,224))
        model.eval()
        super().__init__(model,"resnet152",args)

    
    def get_embedding(self, image):
        with torch.no_grad():
            return self.model(image).cpu()