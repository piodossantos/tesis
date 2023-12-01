from gradio import Blocks, Row, File, JSON, Image, Column, Textbox, Slider, Button, Number, Warning, Error
from experiment_framework import infer
from models.resnet18 import get_model as get_resnet18
from preprocessing.transforms import BASELINE
from clustering.model import clustering_function
from sklearn.cluster import AgglomerativeClustering
from utils import get_longest_intervals, load_video
from functools import lru_cache 
import torch
import cv2
import numpy as np
import json

device = torch.device("cpu")

@lru_cache
def show_sliders(file):
    video = list(load_video(file))
    rgb_video = []
    for frame in video:
        rgb_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    rgb_video = np.array(rgb_video)
    return rgb_video


@lru_cache
def get_clusters(file):
    video = list(load_video(file))
    labels = infer(**{
        "model": get_resnet18(device),
        "preprocessing": BASELINE,
        "grouper_function": clustering_function(AgglomerativeClustering(None, distance_threshold=50)),
        "device": device,
        "stream": video
    })
    rgb_video = []
    for frame in video:
        rgb_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    rgb_video = np.array(rgb_video)
    intervals = get_longest_intervals(labels)
    return rgb_video, intervals


def generate_sliders_row(file, end):
    video = show_sliders(file)
    start = min(end + 1, len(video)-1)
    end = min(end + 60, len(video)-1)
    return video[end], video[start], end, start


def slider_change(number, file):
    video = show_sliders(file)
    index = min(number -1, len(video) -1)
    if number <= len(video):
        return video[index]
    return None


def previous_interval(start, end, file,result): 
    filename=file.split("/")[-1]
    if result:
        parsed_result = json.loads(result)
        if len(parsed_result.get(filename,[])) > 0:
            result = parsed_result
            start, end, _, _ = result[filename].pop()
            result= json.dumps(result, indent=4)
    return result,"","", start, end

def next_interval(file,result, sl_start, sl_end, screen, action):
    video = show_sliders(file)
    if not screen:
        Warning("Falta etiqueta de pantalla")
        return result,action,screen, sl_start, sl_end
    if not action:
        Warning("Falta etiqueta de accion")
        return result,action,screen, sl_start, sl_end
    if sl_start >= sl_end or sl_end > len(video):
        Warning("Intervalo invalido")
        return result,action,screen, sl_start, sl_end

    filename=file.split("/")[-1]
    if not result:
        result= json.dumps({filename:[[sl_start,sl_end,screen,action]]}, indent=4)
    else:
        result = json.loads(result)
        result[filename].append([sl_start,sl_end,screen,action])
        result= json.dumps(result, indent=4)
    start = min(sl_end + 1, len(video)-1)
    end = min(sl_end + 60, len(video)-1)
    return result,"","", start, end


css = '''
.slider{
    align-items:center;
}
'''

def get_tagging_ui():
    tags = []
    with Blocks(css=css) as tag_ui:
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
                    sl_end = Slider(minimum=1, maximum=3000, step=1, label="End", value=0, interactive=True)
            with Row():
                back_button=Button(value="Back")
                next_button=Button(value="Next")

        with Row() as result:
            result = Textbox(interactive=False)

        file.change(generate_sliders_row, [file, sl_end], [im_end, im_start, sl_end, sl_start])
        sl_end.change(slider_change, [sl_end, file], im_end, show_progress=False)
        sl_start.change(slider_change, [sl_start, file], im_start, show_progress=False)
        next_button.click(next_interval,[file, result, sl_start,sl_end,screen,action],[result,action,screen, sl_start, sl_end])
        back_button.click(previous_interval,[sl_start, sl_end, file, result],[result,action,screen, sl_start, sl_end])
    return tag_ui

