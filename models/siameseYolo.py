from torchvision import models
import torch
from models.model import Model
from torch import flatten
from torch import nn
from torch.functional import F
from ultralytics import YOLO

class SiameseYoloNetwork(nn.Module):
    def __init__(self, yolo):
        super(SiameseYoloNetwork, self).__init__()

        # Use the same ResNet model for both branches
        self.yolo = yolo

        # Additional layers for prediction
        self.fc = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 200)
        self.fc3 = nn.Linear(200, 128)


    def forward_one_branch(self, x):
        x = self.yolo.embed(x, verbose=False)
        x = list(map(lambda y: y.unsqueeze(0), x))
        x = torch.cat(x, dim=0)
        x = self.fc(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        return x

    def forward(self, input1, input2):
        # Forward pass for each branch
        x1 = self.forward_one_branch(input1)
        x2 = self.forward_one_branch(input2)
        return x1,x2

class SiameseYoloNet(Model):
    model = None
    @staticmethod
    def get_instance(*args):
        if not SiameseYoloNet.model:
            SiameseYoloNet.model = SiameseYoloNet(*args)
        return SiameseYoloNet.model

    def __init__(self, *args):
        model=YOLO(args[0])
        val_model = SiameseYoloNetwork(model).to(args[1])
        val_model.load_state_dict(torch.load(args[2], map_location=torch.device('cpu')))
        super().__init__(val_model,"siameseYoloNet",args)

    
    def get_embedding(self, image):
        with torch.no_grad():
            return self.model.forward_one_branch(image).cpu()