from torchvision import models
import torch
from models.model import Model


class Resnet18(Model):
    model = None
    @staticmethod
    def get_instance(*args):
        if not Resnet18.model:
            Resnet18.model = Resnet18(*args)
        return Resnet18.model

    def __init__(self, *args):
        model = models.resnet18(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.to(*args)
        #summary(model,(3,224,224))
        model.eval()
        super().__init__(model,"resnet18",args)

    
    def get_embedding(self, image):
        with torch.no_grad():
            return self.model(image).cpu()