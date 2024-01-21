from utils import convertir_pascal_a_yolo
import glob
import os
import random
import shutil
from collections import Counter

OUTPUT = 'vin_yolo'

file_list = glob.glob(os.path.join("All Dataset", "**/*"), recursive=True)
CLASES = ['TextButton', 'Text', 'Image', 'PageIndicator', 'Icon',
          'UpperTaskBar', 'EditText', 'Bottom_Navigation', 'Drawer', 'Toolbar',
          'Card', 'Multi_Tab', 'Spinner', 'CheckedTextView', 'Switch', 'Modal',
          'BackgroundImage', 'Map', 'Remember', 'CheckBox']

acc = []
for index, file in enumerate(file_list):
    if index % 11 == 0:
        print(f"{index/len(file_list)} % avance")
    if '.xml' in file:
        prob = random.uniform(0,1)
        directory = 'vin_yolo/labels/train'
        if prob > 0.9:
            directory = 'vin_yolo/labels/test'
        image_classes = convertir_pascal_a_yolo(file, directory, CLASES)
        acc += image_classes
        image_directory= directory.replace('labels', 'images')
        image_file = file.replace('Annotations', 'JPEGImages').replace('.xml', '.jpg')
        shutil.copy(image_file, image_directory)

print(acc)
dicc = Counter(acc)

print(dicc)