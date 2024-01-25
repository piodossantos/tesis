from preprocessing.transforms import BASELINE, YOLO_BASELINE
from sklearn.cluster import AgglomerativeClustering
from clustering.similarity import SimilarityClustering
from clustering.model import clustering_function
from models.resnet18 import Resnet18
from models.yolov8 import YOLOV8

MODELS_ENUM = {
    'RESNET18': Resnet18,
    'YOLOV8': YOLOV8
}

PREPROCESSORS = {
    "BASELINE": BASELINE,
    "YOLO_BASELINE": YOLO_BASELINE
}
ALGORITHM = {
    "AGGLOMERATIVE": AgglomerativeClustering,
    "SIMILARITY": SimilarityClustering
}
GROUPER_FUNCTIONS = {
    "CLUSTERING_FUNCTION": clustering_function
}