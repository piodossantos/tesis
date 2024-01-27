import torch
import numpy as np
from preprocessing.transforms import BASELINE
from utils import load_video
from metrics import show_metrics_massive,calculate_mean_var
from validation import VALIDATION_DATASET
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
import os

def infer(device, model, preprocessing, grouper_function,stream,path):
    
    features = []
    filename = f"embeddings/{model.get_name(path)}.npy"
    if os.path.exists(filename):
      features=np.load(filename)
    else:
      print("model name: ",model.get_name(path))
      for frame in stream:
        input_tensor = preprocessing(frame)
        input_batch = input_tensor.unsqueeze(0).to(device)
        output = model.get_embedding(input_batch).numpy().flatten()
        features.append(output)

      features = np.array(features)
      np.save(filename,features)
    labels = grouper_function(features)
    return labels 


def experiment(device, name, model, preprocessing, dataset, grouper_function, evaluation_function, show=False, **kwargs):
    metric_list = defaultdict(list)
    for path,stream in dataset.items():
      tag = VALIDATION_DATASET[path]
      labels = infer(device, model, preprocessing,  grouper_function,stream,path)
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
