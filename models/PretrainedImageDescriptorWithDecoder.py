from torchvision import models
import torch
from models.model import Model
from transformers import Pix2StructProcessor,Pix2StructForConditionalGeneration
from torch import nn
from PIL import Image
import numpy as np
class PretrainedImageDescriptorWithDecoder(Model):
    model = None
    @staticmethod
    def get_instance(*args):
        if not PretrainedImageDescriptorWithDecoder.model:
            PretrainedImageDescriptorWithDecoder.model = PretrainedImageDescriptorWithDecoder(*args)
        return PretrainedImageDescriptorWithDecoder.model

    def __init__(self, *args):
        self.device=args[0]
        self.text=args[2]
        self.model = Pix2StructProcessor.from_pretrained(args[1])
        self.decoder =Pix2StructForConditionalGeneration.from_pretrained(args[1]).to(self.device)
        self.x=args[3]
        self.last_vector=None
        super().__init__(self.model,"pretrained image descriptor decoder",args)

    def text_to_mean_vector(self,text, vector_size=300):
        words = text.split()
        valid_vectors = [self.x[word] for word in words if word in self.x]
        
        if valid_vectors:
            # Calcular la media de los vectores válidos
            mean_vector = np.sum(valid_vectors, axis=0)
        else:
            # Si no hay vectores válidos, optar por un vector de ceros
            mean_vector = np.zeros(vector_size)
            
        return mean_vector


    def get_embedding(self, image):
        # image only
        image = image.squeeze(0)

        image = image.cpu().numpy()
        image=image*255
        image = image.astype(np.uint8).transpose(1,2,0)
        image=Image.fromarray(image)
        inputs = self.model(images=image, text=self.text,return_tensors="pt").to(self.device)
        


        if not self.last_vector:


            predictions = self.decoder.generate(**inputs)
            output = self.model.decode(predictions[0], skip_special_tokens=True)

            mean_vector = self.text_to_mean_vector(output)

            self.result = mean_vector
            self.last_vector = inputs.flattened_patches.flatten()
            parecidos = 1
            
        else: 
            
            ##Optimizamos
            # Calcular la similitud del coseno
            similitud_coseno = np.dot(inputs.flattened_patches.flatten(), self.last_vector) / (np.linalg.norm(inputs.flattened_patches.flatten()) * np.linalg.norm(self.last_vector))

            # Definir el umbral para la similitud del coseno
            umbral_similitud = 0.95  # Tu valor de umbral

            # Comparar con el umbral
            parecidos = similitud_coseno > umbral_similitud
            
            if parecidos:
                mean_vector= self.result
            else:    
                predictions = self.decoder.generate(**inputs)
                output = self.model.decode(predictions[0], skip_special_tokens=True)

                mean_vector = self.text_to_mean_vector(output)

                self.last_vector = inputs
                self.result=mean_vector
            print(mean_vector)
            
        return mean_vector






