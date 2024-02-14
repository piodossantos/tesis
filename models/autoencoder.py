from ultralytics import YOLO
from models.model import Model
from torch import flatten, reshape
import torch.nn.functional as F
from torch import nn
import torch
from PIL import Image, ImageDraw
import numpy as np
from utils import draw_mask



class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.norm1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 5)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, 3)
        self.norm4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 8, 1)
        self.norm5 = nn.BatchNorm2d(8)
        self.conv6 = nn.Conv2d(8, 4, 1)
        self.norm6 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(144, 64)

        self.unfc1 = nn.Linear(64, 144)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.unconv1 = nn.ConvTranspose2d(4, 8, 1)
        self.unconv2 = nn.ConvTranspose2d(8, 16, 1)
        self.unconv3 = nn.ConvTranspose2d(16, 32, 3)
        self.unconv4 = nn.ConvTranspose2d(32, 64, 3)
        self.unconv5 = nn.ConvTranspose2d(64, 128, 5)
        self.unconv6 = nn.ConvTranspose2d(128, 3, 5)
        self.finalUpsample = nn.UpsamplingNearest2d((252, 252))

    def encode(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.norm5(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.conv6(x)
        x = self.norm6(x)
        x = F.leaky_relu(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        return x

    def decode(self, x):
        x = F.elu(self.unfc1(x))
        x = reshape(x, (-1, 4, 6, 6))
        x = F.elu(self.unconv1(self.upsample(x)))
        x = F.elu(self.unconv2(self.upsample(x)))
        x = F.elu(self.unconv3(self.upsample(x)))
        x = F.elu(self.unconv4(self.upsample(x)))
        x = F.elu(self.unconv5(self.upsample(x)))
        x = F.sigmoid(self.unconv6(self.finalUpsample(x)))
        return x


    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class Encoder(Model):
    model = None

    @staticmethod
    def get_instance(*args):
        if not Encoder.model:
            Encoder.model = Encoder(*args)
        return Encoder.model

    def __init__(self, *args):
        model = YOLO(args[0])
        self.model = model
        encoder = AutoEncoder()
        encoder.load_state_dict(torch.load(args[1], map_location=torch.device('cpu')))
        encoder.to(args[2])
        encoder.eval()
        self.encoder = encoder
        self.device = args[2]
        super().__init__(model,"encoder",args)
    
    def get_embedding(self, image):
        pred = self.model.predict(image, verbose=False)[0].cpu()
        image = draw_mask(pred.boxes.cls, pred.boxes.xyxyn * 256).to(self.device)
        encoded = self.encoder.encode(image).detach().cpu()
        return encoded