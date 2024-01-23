from abc import ABC


class Model(ABC):

    def get_instance(*args):
        pass
    
    def get_embedding(self, image):
        pass
