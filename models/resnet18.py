from torchvision import models
from torchsummary import summary
import torch


def get_model(device):
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    #summary(model,(3,224,224))
    model.eval()

    return model