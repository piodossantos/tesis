from contextlib import contextmanager
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
from bs4 import BeautifulSoup
import json
from PIL import Image, ImageDraw
import numpy as np
import torch

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
  def dataset_loader():
    result = {}
    for path in path_list:
      result[path] = load_video(f'data/{path}')
    return result
  return dataset_loader

def convertir_pascal_a_yolo(archivo_pascal, directorio_salida, clases):
    image_classes = []
    with open(archivo_pascal, "r") as pascal_file:
        soup = BeautifulSoup(pascal_file, "xml")
        size = soup.find("size")
        w=float(size.find("width").text)
        h=float(size.find("height").text)
        lines_yolo = []
        for object_elem in soup.find_all("object"):
            class_name = object_elem.find("name").text
            xmin = float(object_elem.find("xmin").text)
            ymin = float(object_elem.find("ymin").text)
            xmax = float(object_elem.find("xmax").text)
            ymax = float(object_elem.find("ymax").text)
            
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            width = (xmax - xmin) / 1.0
            height = (ymax - ymin) / 1.0
            image_classes.append(class_name)
            clase_id = clases.index(class_name)
            
            line_yolo = f"{clase_id} {x_center / w} {y_center/h} {width/w} {height/h}\n"
            lines_yolo.append(line_yolo)
    
    nombre_salida = os.path.splitext(os.path.basename(archivo_pascal))[0] + ".txt"
    with open(os.path.join(directorio_salida, nombre_salida), "w") as yolo_file:
        for line_yolo in lines_yolo:
            print("linea", line_yolo)
            if line_yolo:
              yolo_file.write(line_yolo)
    return image_classes

CLASSES = ['TextButton', 'Text', 'Image', 'PageIndicator', 'Icon',
          'UpperTaskBar', 'EditText', 'Bottom_Navigation', 'Drawer', 'Toolbar',
          'Card', 'Multi_Tab', 'Spinner', 'CheckedTextView', 'Switch', 'Modal',
          'BackgroundImage', 'Map', 'Remember', 'CheckBox']

def recurse(element,image_classes,clases,lines_yolo,w,h):
    if isinstance(element, dict):
        if "componentLabel" in element:
            class_name = element["componentLabel"].upper()
            bounds = element["bounds"]
            xmin,ymin, xmax,ymax = bounds
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            width = (xmax - xmin) / 1.0
            height = (ymax - ymin) / 1.0
            image_classes.append(class_name)
            clase_id = clases.index(class_name)
            
            line_yolo = f"{clase_id} {x_center / w} {y_center/h} {width/w} {height/h}\n"
            lines_yolo.append(line_yolo)
            #print(lines_yolo)
        for _, value in element.items():
            if isinstance(value, (dict, list)):
                recurse(value,image_classes,clases,lines_yolo,w,h)
    elif isinstance(element, list):
        for item in element:
            recurse(item,image_classes,clases,lines_yolo,w,h)

def convertir_json_a_yolo(archivo_json, directorio_salida, clases):

    image_classes = []
    with open(archivo_json, "r") as json_file:
        #print(json_file)
        elem = json.load(json_file)
        #print(elem)
        bounds = elem["bounds"]
        w,h = bounds[2],bounds[3]
       # print(w,h)
        lines_yolo = []
        recurse(elem,image_classes,clases,lines_yolo,w,h)
        #print("linesYOLO",lines_yolo)
    #     for object_elem in soup.find_all("object"):
    #         class_name = object_elem.find("name").text
    #         xmin = float(object_elem.find("xmin").text)
    #         ymin = float(object_elem.find("ymin").text)
    #         xmax = float(object_elem.find("xmax").text)
    #         ymax = float(object_elem.find("ymax").text)
            
    #         x_center = (xmin + xmax) / 2.0
    #         y_center = (ymin + ymax) / 2.0
    #         width = (xmax - xmin) / 1.0
    #         height = (ymax - ymin) / 1.0
    #         image_classes.append(class_name)
    #         clase_id = clases.index(class_name)
            
    #         line_yolo = f"{clase_id} {x_center / w} {y_center/h} {width/w} {height/h}\n"
    #         lines_yolo.append(line_yolo)
    
    nombre_salida = os.path.splitext(os.path.basename(archivo_json))[0] + ".txt"
    with open(os.path.join(directorio_salida, nombre_salida), "w") as yolo_file:
        for line_yolo in lines_yolo:
            #xxprint("linea", line_yolo)
            if line_yolo:
              yolo_file.write(line_yolo)
    return image_classes

def draw_mask(class_tensor, boxes_tensor):
  classes = class_tensor.tolist()
  boxes = boxes_tensor.tolist()
  class_names = map(lambda i: CLASSES[int(i)], classes)
  results = zip(boxes, class_names)
  images = []
  elems = []
  image = Image.new('RGB', (256, 256), (255,255,255))
  draw = ImageDraw.Draw(image)
  for box, class_name in results:
      x0, x1, y0, y1 = box
      if class_name == "Text":
        elems.append((x0, x1, y0, y1, (255, 0, 0)))
      if class_name == "TextButton" or class_name == "Icon" or class_name == "Switch" or class_name == "CheckBox":
        elems.append((x0, x1, y0, y1, (0, 255, 0)))
      if class_name == "Image" or class_name == "BackgroundImage":
        images.append((x0, x1, y0, y1, (0, 0, 255)))
      if class_name == "EditText":
        elems.append((x0, x1, y0, y1, (0, 0, 0)))
  for (x0, x1, y0, y1, color) in images:
    draw.rectangle(
        ((x0, x1),
        (y0, y1)
    ), color)
  for (x0, x1, y0, y1, color) in elems:
      draw.rectangle(
          ((x0, x1),
          (y0, y1)
      ), color)
  tensor = torch.Tensor(np.array(image))
  tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255
  return tensor


def get_label_tsv(dataset):
    data = ["frame\tvideo\tinterval\n"]
    for video, intervals in dataset.items():
      video_frames = load_video(f'data/{video}')
      for index, _ in enumerate(video_frames):
        tag_index = index + 1
        interval = list(filter(lambda x:  x[0] <= tag_index <= x[1], intervals))
        if len(interval):
          class_name = interval[0][2]
        else:
          class_name = "NONE"
        data.append("\t".join([str(tag_index), video, class_name]) + '\n')
    with open(f'embedding_tags.tsv', 'w') as f:
      f.writelines(data)
    return data
