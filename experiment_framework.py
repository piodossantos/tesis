import torch
import numpy as np
from preprocessing.transforms import BASELINE
from utils import load_video
from metrics import show_metrics_massive,calculate_mean_var
from validation import VALIDATION_DATASET
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2

def infer(device, model, preprocessing, grouper_function,stream):
    
    features = []
   
    for frame in stream:
        input_tensor = preprocessing(frame)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.show()
        input_batch = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.embed(input_batch)
        print(output)
        features.append(output.cpu().numpy().flatten())

    features = np.array(features)



    labels = grouper_function(features)
    return labels 


def experiment(device, name, model, preprocessing, dataset, grouper_function, evaluation_function, show=False, **kwargs):
    metric_list = defaultdict(list)
    for path,stream in dataset.items():
      tag = VALIDATION_DATASET[path]
      labels = infer(device, model, preprocessing,  grouper_function,stream)
      metrics = evaluation_function(labels, tag)
      metric_list["precision"].append(metrics.precision.mean)
      metric_list["recall"].append(metrics.recall.mean)
      metric_list["accuracy"].append(metrics.accuracy.mean)
      metric_list["f1"].append(metrics.f1.mean)
      if(show):
        show_metrics_massive(name+" "+path, metrics)
    result  = calculate_mean_var( metric_list["precision"], metric_list["recall"],  metric_list["f1"], metric_list["accuracy"])
    if(show):
      show_metrics_massive(name+" AVG", result)
    return result
