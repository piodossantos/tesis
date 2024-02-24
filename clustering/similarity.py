import numpy as np
from torch.functional import F
import torch

class SimilarityClustering:
    def __init__(self, threshold, distance='cosine'):
        self.threshold = threshold
        self.labels_ = None
        self.distance = distance

    def fit(self, features):
        if self.distance == 'cosine':
            distance = F.cosine_similarity
        if self.distance == 'euclidean':
            distance = F.pairwise_distance

        cosine_similarities = [distance(torch.Tensor([features[i]]), torch.Tensor([features[i + 1]]))
                                        for i in range(len(features) - 1)]
        cosine_similarities = list(map(float, cosine_similarities))

        cosine_similarities = ([1.0] if self.distance == 'cosine' else  [0.0]) + cosine_similarities

        self.labels_ = label_clusters(cosine_similarities, self.threshold, self.distance)

def label_clusters(cosine_similarities, threshold, distance):
    clusters = np.zeros(len(cosine_similarities), dtype=int)
    current_cluster = 0
    for i in range(0, len(cosine_similarities)):
        #print(i,cosine_similarities[i])
        if distance == 'euclidean':
            if cosine_similarities[i] > threshold:
                current_cluster += 1
        if distance == 'cosine':
            if cosine_similarities[i] < threshold:
                current_cluster += 1
        
        clusters[i] = current_cluster
    return clusters

def clustering_function(model):
    def get_labels(features):
        model.fit(features)
        labels = model.labels_
        return labels
    return get_labels