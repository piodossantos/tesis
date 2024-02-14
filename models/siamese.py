from torchvision import models
import torch
from models.model import Model
from torch import flatten
from torch import nn
from torch.functional import F

class SiameseNetwork(nn.Module):
    def __init__(self, resnet_model):
        super(SiameseNetwork, self).__init__()

        # Use the same ResNet model for both branches
        self.resnet_branch1 = resnet_model

        # Additional layers for prediction
        self.fc = nn.Linear(512, 400)
        self.fc2 = nn.Linear(400, 128)


    def forward_one_branch(self, x):
        x = self.resnet_branch1(x)
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


    def forward_one_branch(self, x):
        x = self.resnet_branch1(x)
        # Add any additional processing if needed
        x = flatten(x, 1)
        # x = self.dropout(x)
        x = F.leaky_relu(self.fc(x))
        x = F.leaky_relu(self.fc2(x))
        return x

    def forward(self, input1, input2):
        # Forward pass for each branch
        x1 = self.forward_one_branch(input1)
        x2 = self.forward_one_branch(input2)
        return x1,x2

class SiameseResnet18(Model):
    model = None
    @staticmethod
    def get_instance(*args):
        if not SiameseResnet18.model:
            SiameseResnet18.model = SiameseResnet18(*args)
        return SiameseResnet18.model

    def __init__(self, *args):
        model = models.resnet18(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        #summary(model,(3,224,224))
        model.eval()
        val_model = SiameseNetwork(model).to(args[0])
        val_model.load_state_dict(torch.load(args[1], map_location=torch.device('cpu')))
        self.model = val_model.eval()
        super().__init__(val_model,"siameseresnet18",args)

    
    def get_embedding(self, image):
        with torch.no_grad():
            return self.model.forward_one_branch(image).cpu()