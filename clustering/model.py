def clustering_function(model):
    def get_labels(features):
        model.fit(features)
        labels = model.labels_
        return labels
    return get_labels