import glob
import os
from PIL import Image, ImageDraw

OUTPUT = 'masks'

VERSION ="1"


FOLDER_NAME=f'{OUTPUT}_{VERSION}'

if not os.path.exists(FOLDER_NAME):
    os.mkdir(FOLDER_NAME)

file_list = glob.glob(os.path.join("vin_yolo_2", "**/*.txt"), recursive=True)
CLASES = ['TextButton', 'Text', 'Image', 'PageIndicator', 'Icon',
          'UpperTaskBar', 'EditText', 'Bottom_Navigation', 'Drawer', 'Toolbar',
          'Card', 'Multi_Tab', 'Spinner', 'CheckedTextView', 'Switch', 'Modal',
          'BackgroundImage', 'Map', 'Remember', 'CheckBox']

for file in file_list:
    image = Image.new('RGB', (256, 256), (255,255,255))
    draw = ImageDraw.Draw(image)
    images = []
    elems = []
    with open(file, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            classname, x, y, w, h = line.split(" ")
            classname = CLASES[int(classname)]
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            if classname == "Text":
                elems.append(((x - w)*256, (y - h)*256, (x + w)*256, (y + h)*256, (255, 0, 0)))
            if classname == "TextButton" or classname == "Icon" or classname == "Switch" or classname == "CheckBox":
                elems.append(((x - w)*256, (y - h)*256, (x + w)*256, (y + h)*256, (0, 255, 0)))
            if classname == "Image" or classname == "BackgroundImage":
                images.append(((x - w)*256, (y - h)*256, (x + w)*256, (y + h)*256, (0, 0, 255)))
            if classname == "EditText":
                elems.append(((x - w)*256, (y - h)*256, (x + w)*256, (y + h)*256, (0, 0, 0)))
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
    
    image_name = file.split("/")[-1]
    image_name = image_name.replace(".txt", '.jpg')
    image.save(f'{FOLDER_NAME}/{image_name}')

