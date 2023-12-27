import numpy as np

class SimilarityClustering:
    def __init__(self, threshold):
        self.threshold = threshold
        self.labels_ = None

    def fit(self, features):
        cosine_similarities = np.array([cosine_similarity(features[i], features[i + 1])
                                        for i in range(len(features) - 1)])

        cosine_similarities = np.insert(cosine_similarities, 0, cosine_similarity(features[0], features[0]))

        self.labels_ = label_clusters(cosine_similarities, self.threshold)

cosine_similarity = lambda vec1, vec2: np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def label_clusters(cosine_similarities, threshold):
    clusters = np.zeros(len(cosine_similarities), dtype=int)
    current_cluster = 0
    for i in range(1, len(cosine_similarities)):
        #print(i,cosine_similarities[i])

        if cosine_similarities[i] < threshold:
            current_cluster += 1
        clusters[i] = current_cluster
    #print(clusters)
    return clusters

def clustering_function(model):
    def get_labels(features):
        model.fit(features)
        labels = model.labels_
        return labels
    return get_labels