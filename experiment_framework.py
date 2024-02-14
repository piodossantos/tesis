import numpy as np
from preprocessing.transforms import BASELINE
from utils import load_video
from metrics import show_metrics_massive,calculate_mean_var
from validation import VALIDATION_DATASET
from collections import defaultdict
import matplotlib.pyplot as plt
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
    return labels, features


def experiment(device, name, model, preprocessing, dataset, grouper_function, evaluation_function, show=False, **kwargs):
    metric_list = defaultdict(list)
    all_video_embedings = []
    for path,stream in dataset.items():
      tag = VALIDATION_DATASET[path]
      labels, embeddings = infer(device, model, preprocessing,  grouper_function,stream,path)
      all_video_embedings += list(embeddings)
      metrics = evaluation_function(labels, tag)
      metric_list["precision"].append(metrics.precision.mean)
      metric_list["recall"].append(metrics.recall.mean)
      metric_list["accuracy"].append(metrics.accuracy.mean)
      metric_list["f1"].append(metrics.f1.mean)
      if(show):
        show_metrics_massive(name+" "+path, metrics)
    filename = f"embeddings/{model.name}.tsv"
    if not os.path.exists(filename):
      tsv = []
      for embedding in all_video_embedings:
        tsv.append("\t".join(map(str, embedding.tolist()))+ "\n")
      with open(filename, 'w') as f:
          f.writelines(tsv)
    result  = calculate_mean_var( metric_list["precision"], metric_list["recall"],  metric_list["f1"], metric_list["accuracy"])
    if(show):
      show_metrics_massive(name+" AVG", result)
    return result
