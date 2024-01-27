from utils import convertir_json_a_yolo
import glob
import os
import random
import shutil
from collections import Counter
from ultralytics.data.utils import compress_one_image
from PIL import Image
OUTPUT = 'rico'

VERSION ="1"

FOLDER_NAME=f'{OUTPUT}_{VERSION}'
random.seed(41)
if not os.path.exists(FOLDER_NAME):
    os.mkdir(FOLDER_NAME)
    os.mkdir(f'{FOLDER_NAME}/images')
    os.mkdir(f'{FOLDER_NAME}/labels')
    os.mkdir(f'{FOLDER_NAME}/images/train')
    os.mkdir(f'{FOLDER_NAME}/labels/train')
    os.mkdir(f'{FOLDER_NAME}/images/test')
    os.mkdir(f'{FOLDER_NAME}/labels/test')    
    os.mkdir(f'{FOLDER_NAME}/images/val')
    os.mkdir(f'{FOLDER_NAME}/labels/val')

file_list = glob.glob(os.path.join("semantic_annotations", "**/*"), recursive=True)
CLASES = [
    'ADVERTISEMENT',
    'BACKGROUND IMAGE',
    'BOTTOM NAVIGATION',
    'BUTTON BAR',
    'CARD',
    'CHECKBOX',
    'DATE PICKER',
    'DRAWER',
    'ICON',
    'IMAGE',
    'INPUT',
    'LIST ITEM',
    'MAP VIEW',
    'MODAL',
    'MULTI-TAB',
    'NUMBER STEPPER',
    'ON/OFF SWITCH',
    'PAGER INDICATOR',
    'RADIO BUTTON',
    'SLIDER',
    'TEXT',
    'TEXT BUTTON',
    'TOOLBAR',
    'VIDEO',
    'WEB VIEW'
]
acc = []
for index, file in enumerate(file_list):
    if index % 25 == 0:
        print(f"{index/len(file_list)*100} % avance")
    if '.json' in file:
        sanitize_filename= file.replace('semantic_annotations/','').replace(".json",".txt")
        #print(sanitize_filename)
        if os.path.exists(f'rico_1/labels/test/{sanitize_filename}') or os.path.exists(f'rico_1/labels/train/{sanitize_filename}')  or os.path.exists(f'rico_1/labels/val/{sanitize_filename}') :
            pass
        else:
            # print(file)
            prob = random.uniform(0,1)
            directory = f'{FOLDER_NAME}/labels/train'
            if prob > 0.9:
                directory = f'{FOLDER_NAME}/labels/test'
            elif prob>0.7 and prob<=0.9:
                directory = f'{FOLDER_NAME}/labels/val'
            image_classes = convertir_json_a_yolo(file, directory, CLASES)
            acc += image_classes
            image_directory= directory.replace('labels', 'images')
            image_file = file.replace('Annotations', 'JPEGImages').replace('.json', '.jpg').replace("semantic_annotations","combined")
        # print(image_file,image_directory)
            # compress_one_image(image_file)
            
            #shutil.copy(image_file, image_directory)
            img = Image.open(image_file)
            img=img.resize((640,640))
            file_aux=f'{image_directory}/{image_file}'.replace('combined/','')
            img.save(file_aux)
            compress_one_image(file_aux)

print(acc)
dicc = Counter(acc)

print(dicc)