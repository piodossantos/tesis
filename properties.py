from preprocessing.transforms import BASELINE, YOLO_BASELINE
from sklearn.cluster import AgglomerativeClustering
from clustering.similarity import SimilarityClustering
from sklearn.cluster import DBSCAN
from clustering.model import clustering_function
from models.resnet18 import Resnet18
from models.yolov8 import YOLOV8
from models.siamese import SiameseResnet18
from models.autoencoder import Encoder
from models.yolores import YoloRes
from models.mobileNetV2 import MobileNetV2
from models.VGG19 import VGG19
from models.resnet152 import Resnet152
from models.siameseMobileNet import SiameseMobileNet
from models.siameseYolo import SiameseYoloNet
from models.PretrainedImageDescriptor import PretrainedImageDescriptor

MODELS_ENUM = {
    'RESNET18': Resnet18,
    'YOLOV8': YOLOV8,
    'SIAMESE_RESNET18': SiameseResnet18,
    'SIAMESE_YOLO':SiameseYoloNet,
    'SIAMESE_MOBILENET':SiameseMobileNet,
    'Encoder': Encoder,
    'YOLORES': YoloRes,
    'MOBILENETV2': MobileNetV2,
    'VGG19': VGG19,
    'RESNET152': Resnet152,
    'PRETRAINED_IMAGE_DESCRIPTOR': PretrainedImageDescriptor,
}

PREPROCESSORS = {
    "BASELINE": BASELINE,
    "YOLO_BASELINE": YOLO_BASELINE
}
ALGORITHM = {
    "AGGLOMERATIVE": AgglomerativeClustering,
    "SIMILARITY": SimilarityClustering,
    "DBSCAN":DBSCAN
}
GROUPER_FUNCTIONS = {
    "CLUSTERING_FUNCTION": clustering_function
}