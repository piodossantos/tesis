import torch
import numpy as np
from preprocessing.transforms import BASELINE
from utils import load_video
from metrics import show_metrics_massive
from validation import VALIDATION_DATASET


def experiment(device, name, model, preprocessing, path, grouper_function, evaluation_function, show=False):
    features = []
    stream = load_video(path)
    for frame in stream:
        input_tensor = preprocessing(frame)
        input_batch = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_batch)
        features.append(output.cpu().numpy().flatten())

    features = np.array(features)
    labels = grouper_function(features)
    tag = VALIDATION_DATASET[path]
    metrics = evaluation_function(labels, tag)
    if(show):
       show_metrics_massive(name, metrics)
    return metrics, labels