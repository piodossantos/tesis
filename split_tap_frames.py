import os
from tapAndScroll import TRAIN_TAPS, TEST_TAPS
from utils import load_video
from collections import defaultdict
from PIL import Image
import cv2
import uuid
import hashlib


if not os.path.exists('taps'):
    os.mkdir('taps')

def is_frame_from_tap_interval(frame_number, interval):
    start, end, class_name = interval
    return class_name.upper() == "TAP" and start <= frame_number <= end

def separate_frames(dataset_name, file_name):
    data = ['filepath,label\n']
    data_dict = defaultdict(list)
    for video, intervals in dataset_name.items():
        video_frames = load_video(f'data/{video}')
        for index, frame in enumerate(video_frames):
            tag_index = index + 1
            taps = list(filter(lambda x: is_frame_from_tap_interval(tag_index, x), intervals))
            if len(taps):
                class_name = "TAP"
                data_dict["TAP"].append(tag_index)
            else:
                class_name = "NO_TAP"
                data_dict["NO_TAP"].append(tag_index)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img.thumbnail([640, 640],Image.LANCZOS)
            frame.flags.writeable = False
            img_name = f'{hashlib.sha256(frame.data).hexdigest()}.jpg'
            data.append(",".join([img_name, class_name]) + '\n')
            img.save(f'taps/{img_name}')

    with open(f'taps/{file_name}', 'w') as f:
        f.writelines(data)


separate_frames(TRAIN_TAPS, 'train.csv')
separate_frames(TEST_TAPS, 'test.csv')

