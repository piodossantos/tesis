from functools import cache
from ultralytics import YOLO


@cache
def get_model(device):
    model = YOLO('/Users/mmandirola/tesis/models/best.pt')
    return model