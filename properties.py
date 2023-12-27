from models.resnet18 import get_model as resnet18
from preprocessing.transforms import BASELINE
from sklearn.cluster import AgglomerativeClustering
from clustering.similarity import SimilarityClustering
from clustering.model import clustering_function


MODELS_ENUM = {
    'RESNET18': resnet18,
}
PREPROCESSORS = {
    "BASELINE": BASELINE
}
ALGORITHM = {
    "AGGLOMERATIVE": AgglomerativeClustering,
    "SIMILARITY": SimilarityClustering
}
GROUPER_FUNCTIONS = {
    "CLUSTERING_FUNCTION": clustering_function
}