import numpy as np
def clustering_function(model):
    def get_labels(features):
        model.fit(features)
        labels = model.labels_
        # print(model.distances_)
        #distances=model.distances_

        # print("n_clusters",model.n_clusters_)
        #print("el masimo es",np.mean(distances),max(distances))
        return labels
    return get_labels