from torchvision import models
import torch
from models.model import Model
from torch import nn
from PIL import Image
import numpy as np
class PretrainedImageDescriptor(Model):
    model = None
    @staticmethod
    def get_instance(*args):
        if not PretrainedImageDescriptor.model:
            PretrainedImageDescriptor.model = PretrainedImageDescriptor(*args)
        return PretrainedImageDescriptor.model

    def __init__(self, *args):
        self.device=args[0]
        self.text=args[2]
        self.model = Pix2StructProcessor.from_pretrained(args[1])
        super().__init__(self.model,"pretrained image descriptor",args)

    
    def get_embedding(self, image):
        # image only
        image = image.squeeze(0)

        image = image.cpu().numpy()
        image=image*255
        image = image.astype(np.uint8).transpose(1,2,0)
        image=Image.fromarray(image)
        inputs = self.model(images=image, text=self.text,return_tensors="pt").to(self.device)
        inputs = inputs.flattened_patches
        print(inputs.shape)
        inputs= nn.AdaptiveAvgPool2d(output_size=(25, 25))(inputs)
        print(inputs.shape)
        inputs= inputs.flatten()
        print(inputs.shape)
        return inputs.cpu()
