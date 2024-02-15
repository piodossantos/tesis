from utils import convertir_json_a_yolo
import glob
import os
import random
import shutil
from collections import Counter
from ultralytics.data.utils import compress_one_image
from PIL import Image
import datetime
import cv2
OUTPUT = 'rico'

VERSION ="5"

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
start = datetime.datetime.now()
for index, file in enumerate(file_list):
    if index>0 and index % 70 == 0:
        avance= index/len(file_list)
        diferencia = datetime.datetime.now() - start
        tiempo_restante = ((1-avance) /avance ) * diferencia.total_seconds()
        tiempo_restante = datetime.timedelta(seconds=tiempo_restante)
        minutos = (tiempo_restante.seconds % 3600) // 60
        segundos = tiempo_restante.seconds % 60
        print(f"{avance*100} % avance, tiempo restante:{minutos}mm{segundos}ss")
    if '.json' in file:
        sanitize_filename= file.replace('semantic_annotations/','').replace(".json",".txt")
        #print(sanitize_filename)
        if os.path.exists(f'{FOLDER_NAME}/labels/test/{sanitize_filename}') or os.path.exists(f'{FOLDER_NAME}/labels/train/{sanitize_filename}')  or os.path.exists(f'{FOLDER_NAME}/labels/val/{sanitize_filename}') :
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
            #compress_one_image(image_file)
            
            shutil.copy(image_file, image_directory)
            #img = Image.open(image_file)
            #img=img.resize((256,256))
            file_aux=f'{image_directory}/{image_file}'.replace('combined/','')
            #img.save(file_aux,'JPEG')
            # #compress_one_image(file_aux)

            #img = cv2.imread(image_file)

        # Check if the image was loaded successfully
            # Resize the image to 256x256 pixels
            #img_resized = cv2.resize(img, (10, 10))

            # Specify the output file path (e.g., as a PNG image)

            # Save the resized image
            #cv2.imwrite(file_aux, img_resized,[cv2.IMWRITE_JPEG_QUALITY, 50])
print(acc)
dicc = Counter(acc)

print(dicc)