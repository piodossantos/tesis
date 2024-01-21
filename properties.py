from models.resnet18 import get_model as resnet18
from models.yolov8 import get_model as yolov8
from preprocessing.transforms import BASELINE, YOLO_BASELINE
from sklearn.cluster import AgglomerativeClustering
from clustering.similarity import SimilarityClustering
from clustering.model import clustering_function


MODELS_ENUM = {
    'RESNET18': resnet18,
    'YOLOV8': yolov8
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