import numpy as np
import metrics


def tag_cluster(cluster_labels, tags,steps):


  tagged_clusters = {}
  for _ in range(steps):
    shuffle_tags=np.random.permutation(tags)
    for start, end, screen, action in shuffle_tags:
      frames = cluster_labels[int(start):int(end)]
      frames = list(filter( lambda x: x not in tagged_clusters.keys(), frames))
      values, counter= np.unique(frames, return_counts=True)
      if(len(counter)>0):
        tagged_clusters[values[np.argmax(counter)]] = screen
  result=[]
  for frames in cluster_labels:
    result.append(tagged_clusters.get(frames,"NO_CLASS"))
  return result


def eval_cluster(cluster_labels, tags,steps):
  y_pred = tag_cluster(cluster_labels, tags,steps)
  return metrics.eval_prediction(y_pred,tags,steps)


def eval_massive_cluster(cluster_labels, tags,steps,epochs):
  precision_list, recall_list, f1_list, accuracy_list =[],[],[],[]
  for _ in range(epochs):
    _, precision, recall, f1, accuracy = eval_cluster(cluster_labels, tags,steps)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    accuracy_list.append(accuracy)
  return metrics.calculate_mean_var(precision_list,recall_list, f1_list, accuracy_list)