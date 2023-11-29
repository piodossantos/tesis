from gradio import Blocks, Row, File, JSON, Image, Column, Textbox, Slider, Button, Number, Warning, Error
from experiment_framework import infer
from models.resnet18 import get_model as get_resnet18
from preprocessing.transforms import BASELINE
from clustering.model import clustering_function
from sklearn.cluster import AgglomerativeClustering
from utils import get_longest_intervals
from functools import lru_cache 
import torch
import cv2
import numpy as np
import json

device = torch.device("cpu")

@lru_cache
def get_clusters(file):
    labels, video = infer(**{
        "model": get_resnet18(device),
        "preprocessing": BASELINE,
        "path": file,
        "grouper_function": clustering_function(AgglomerativeClustering(None, distance_threshold=50)),
        "device": device
    })
    rgb_video = []
    for frame in video:
        rgb_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    rgb_video = np.array(rgb_video)
    intervals = get_longest_intervals(labels)
    return rgb_video, intervals


def generate_sliders_row(file, index):
    video, intervals = get_clusters(file)
    if index >= len(intervals):
        Warning("No hay mas clusters detectados, puede seguir etiquetando")
        return video[len(video)-1], video[0], len(video) - 1, 0
    start, end, _ = intervals[int(index)]
    return video[end-1], video[start-1], end, start


def slider_change(number, file):
    video, _ = get_clusters(file)
    index = min(number -1, len(video) -1)
    if number <= len(video):
        return video[index]
    return None
def previous_interval(index, file,result): 
    filename=file.split("/")[-1]
    if result:
        parsed_result = json.loads(result)
        if len(parsed_result.get(filename,[])) > 0:
            result = parsed_result
            result[filename].pop()
            result= json.dumps(result, indent=4)
    return result,"","",max(int(index)-1,0)

def next_interval(index, file,result, sl_start, sl_end, screen, action):
    video, _ = get_clusters(file)
    if not screen:
        Warning("Falta etiqueta de pantalla")
        return result,action,screen,int(index)
    if not action:
        Warning("Falta etiqueta de accion")
        return result,action,screen,int(index)
    if sl_start >= sl_end or sl_end > len(video):
        Warning("Intervalo invalido")
        return result,action,screen,int(index)

    filename=file.split("/")[-1]
    if not result:
        result= json.dumps({filename:[[sl_start,sl_end,screen,action]]}, indent=4)
    else:
        result = json.loads(result)
        result[filename].append([sl_start,sl_end,screen,action])
        result= json.dumps(result, indent=4)
    return result,"","",int(index)+1
css = '''
.slider{
    align-items:center;
}
'''

def get_tagging_ui():
    tags = []
    with Blocks(css=css) as tag_ui:
        index = Number(value=0, visible=False)
        with Row():
            file = File()
        with Column(visible=True):
            with Row():
                with Column():
                    screen = Textbox(label="Screen")
                    action = Textbox(label="Action")
            with Row():
                with Column(elem_classes="slider"):
                    im_start = Image(interactive=False, width=120)
                    sl_start = Slider(minimum=1, maximum=3000, step=1, label="Start")
                with Column(elem_classes="slider"):
                    im_end = Image(interactive=False, width=120)
                    sl_end = Slider(minimum=1, maximum=3000, step=1, label="End")
            with Row():
                back_button=Button(value="Back")
                next_button=Button(value="Next")

        with Row() as result:
            result = Textbox(interactive=False)

        file.change(generate_sliders_row, [file, index], [im_end, im_start, sl_end, sl_start])
        sl_end.change(slider_change, [sl_end, file], im_end, show_progress=False)
        sl_start.change(slider_change, [sl_start, file], im_start, show_progress=False)
        next_button.click(next_interval,[index,file, result, sl_start,sl_end,screen,action],[result,action,screen,index])
        back_button.click(previous_interval,[index,file, result],[result,action,screen,index])
        index.change(generate_sliders_row, [file, index], [im_end, im_start, sl_end, sl_start])
    return tag_ui

