from torchvision import models
import torch
from models.model import Model
from torch import flatten
from torch import nn
from torch.functional import F

class SiameseMobileNetwork(nn.Module):
    def __init__(self, mobilenet):
        super(SiameseMobileNetwork, self).__init__()

        # Use the same ResNet model for both branches
        self.mobilenet = mobilenet

        # Additional layers for prediction
        self.fc = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 128)


    def forward_one_branch(self, x):
        x = self.mobilenet(x)
        x = F.leaky_relu(x)
        x = flatten(x, 1)
        x = self.fc(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return x

    def forward(self, input1, input2):
        # Forward pass for each branch
        x1 = self.forward_one_branch(input1)
        x2 = self.forward_one_branch(input2)
        return x1,x2

class SiameseMobileNet(Model):
    model = None
    @staticmethod
    def get_instance(*args):
        if not SiameseMobileNet.model:
            SiameseMobileNet.model = SiameseMobileNet(*args)
        return SiameseMobileNet.model

    def __init__(self, *args):
        model = models.mobilenet_v2(pretrained=True)
        val_model = SiameseMobileNetwork(model).to(args[0])
        val_model.load_state_dict(torch.load(args[1], map_location=torch.device('cpu')))
        self.model = val_model.eval()
        super().__init__(val_model,"siameseMobileNet",args)

    
    def get_embedding(self, image):
        with torch.no_grad():
            return self.model.forward_one_branch(image).cpu()