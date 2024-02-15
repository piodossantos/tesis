from torchvision import models
import torch
from models.model import Model


class VGG19(Model):
    model = None
    @staticmethod
    def get_instance(*args):
        if not VGG19.model:
            VGG19.model = VGG19(*args)
        return VGG19.model

    def __init__(self, *args):
        model = models.mobilenet_v2(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.to(*args)
        #summary(model,(3,224,224))
        model.eval()
        super().__init__(model,"VGG19",args)

    
    def get_embedding(self, image):
        with torch.no_grad():
            return self.model(image).cpu()