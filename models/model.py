from abc import ABC
import hashlib

class Model(ABC):

    def get_instance(*args):
        pass
    
    def get_embedding(self, image):
        pass

    def __init__(self,model,name,args):
        self.model = model
        self.name=name
        self.args=args
        self.name=name

    def get_name(self,filename,hyperparams):
        sha = hashlib.sha256()
        sha.update(f'{self.args}{hyperparams}'.encode('utf-8'))
        return f'{self.name}-{filename}-{sha.hexdigest()}'
    
    def get_short_name(self,hyperparams):
        sha = hashlib.sha256()
        sha.update(f'{self.args}{hyperparams}'.encode('utf-8'))
        return f'{self.name}-{sha.hexdigest()}'