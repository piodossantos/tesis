from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from collections import namedtuple
import numpy as np


# Crear una namedtuple para los resultados
MetricsResults = namedtuple('MetricsResults', ['mean', 'std'])


 # Crear una namedtuple para agrupar los resultados
OverallResults = namedtuple('OverallResults', ['precision', 'recall', 'f1', 'accuracy'])

def get_metrics(y_true, y_pred):
    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calcular precisión, recall, F1-score y exactitud
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)

    return conf_matrix, precision, recall, f1, accuracy


def show_metrics(experiment, conf_matrix, precision, recall, f1, accuracy):
    print("Experiment name:",experiment)
    #print(f'Matriz de confusión:\n{conf_matrix}')
   # print(f'Precisión: {precision}')
    #print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
    print(f'Accuracy: {accuracy}')


def eval_prediction(y_pred, tags,steps):
  y_true = []
  for start, end, screen, action in tags:
    y_true+=(end-start+1)*[screen]
  return get_metrics(y_true,y_pred)


def calculate_mean_var(precision_list, recall_list, f1_list, accuracy_list):
  
    # Calcular medias y varianzas para cada métrica usando numpy
    precision_results = MetricsResults(np.mean(precision_list), np.std(precision_list))
    recall_results = MetricsResults(np.mean(recall_list), np.std(recall_list))
    f1_results = MetricsResults(np.mean(f1_list), np.std(f1_list))
    accuracy_results = MetricsResults(np.mean(accuracy_list), np.std(accuracy_list))

   
    return OverallResults(precision=precision_results, recall=recall_results, f1=f1_results, accuracy=accuracy_results)


def show_metrics_massive(experiment, overall_results):
    print()
    print("Experiment name:",experiment)
    print("Precision - Mean: {:.2f}, std: {:.10f}".format(overall_results.precision.mean, overall_results.precision.std))
    print("Recall - Mean: {:.2f}, std: {:.10f}".format(overall_results.recall.mean, overall_results.recall.std))
    print("F1-Score - Mean: {:.2f}, std: {:.10f}".format(overall_results.f1.mean, overall_results.f1.std))
    print("Accuracy - Mean: {:.2f}, std: {:.10f}".format(overall_results.accuracy.mean, overall_results.accuracy.std))
