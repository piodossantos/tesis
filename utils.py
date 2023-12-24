from contextlib import contextmanager
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


@contextmanager
def video_capture_manager(*args, **kwargs):
    try:
        stream = cv2.VideoCapture(*args, **kwargs)
        yield stream
    finally:
        stream.release()


def load_video(path):
    has_next = True
    with video_capture_manager(path) as stream:
        while has_next:
            ret, frame = stream.read()
            if ret:
                yield frame
            has_next = ret


def get_frames_by_label(path, labeled_frames_list):
    frames_by_label = defaultdict(list)
    stream = list(load_video(path))
    for index, frame in enumerate(stream):
        label = labeled_frames_list[index]
        frames_by_label[label].append(frame)
    return frames_by_label


def plot_label_counts(labels_dict):
    """
    Plots a bar chart showing the number of appearances of each label.

    :param labels_dict: A dictionary where keys are labels and values are lists of frames.
    """
    # Counting the number of frames for each label
    label_counts = {label: len(frames) for label, frames in labels_dict.items()}
    print(label_counts)
    # Data for plotting
    labels = list(label_counts.keys())
    counts = list(label_counts.values())

    # Creating the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='pink')
    plt.xlabel('Labels')
    plt.ylabel('Number of Appearances')
    plt.title('Number of Appearances of Labels in Frames')
    plt.show()


def show_separation_sample(images_by_label):
  for label, frames in images_by_label.items():
    frames_per_image = len(frames)
    if(len(frames)>4):
      frames_per_image = 4

    plt.figure(figsize=(10, 5))

    for idx, frame in enumerate(frames[:frames_per_image]):
      plt.subplot(1, frames_per_image, idx+1)  # 1 fila, 2 columnas, posición 2
      plt.axis('off')
      cv2.putText(frame, f'Label: {label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
      plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()


def show_separation_sample(images_by_label):
  for label, frames in images_by_label.items():
    frames_per_image = len(frames)
    if(len(frames)>4):
      frames_per_image = 4

    plt.figure(figsize=(10, 5))

    for idx, frame in enumerate(frames[:frames_per_image]):
      plt.subplot(1, frames_per_image, idx+1)  # 1 fila, 2 columnas, posición 2
      plt.axis('off')
      cv2.putText(frame, f'Label: {label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
      plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()


def show_image_ribbon(images_by_label):
  sorted_labels = sorted(images_by_label.keys())

  plt.figure(figsize=(100, 100))

  for idx, label in enumerate(sorted_labels):

    frames = images_by_label[label]
    plt.subplot(1, len(sorted_labels), idx+ 1)
    plt.axis('off')
    # Calcula la imagen promedio
    # Convierte las imágenes a float para evitar problemas de desbordamiento
    average_frame = np.mean([frame.astype(np.float32) for frame in frames], axis=0)
    average_frame = np.array(np.round(average_frame), dtype=np.uint8)

    # Mostrar la imagen promedio
    plt.imshow(cv2.cvtColor(average_frame, cv2.COLOR_BGR2RGB))
  plt.show()


def get_longest_intervals(labels):
    intervals = defaultdict(list)
    current_interval = (1, 1, labels[0])
    for index, label in enumerate(labels[1:]):
        current_start, current_end, current_value = current_interval
        if(label == current_value):
            current_interval = (current_start, index + 2, label)
        else:
            intervals[current_value].append(current_interval)
            current_interval = (current_end + 1, index + 2, label)
    intervals[current_interval[2]].append(current_interval)
    maximum_intervals = {k:max(v, key=lambda x: x[1] - x[0] + 1) for (k,v) in intervals.items()}
    return sorted(maximum_intervals.values(), key= lambda x: x[0])

def load_dataset(path_list):
  result = {}
  for path in path_list:
    result[path] = np.array(list(load_video(f'data/{path}')))
  return result
