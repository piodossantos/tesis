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