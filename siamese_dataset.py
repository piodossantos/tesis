from utils import load_video
import hashlib
import cv2
from PIL import Image
import csv
from collections import defaultdict
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from numpy import asarray
import json
from functools import reduce


def get_labels_by_hash(dataset):
    data = ['filepath,video,interval,index\n']
    for key, intervals in dataset.items():
        stream = load_video(f'data/{key}')
        for index, frame in enumerate(stream):
            tag_index = index + 1
            find_intervals = list(filter(lambda x: x[0] <= tag_index <= x[1], intervals))
            if len(find_intervals):
                class_name = find_intervals[0][2]
            else:
                class_name = 'NONE'
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img.thumbnail([640, 640],Image.LANCZOS)
            frame.flags.writeable = False
            img_name = f'{hashlib.sha256(frame.data).hexdigest()}.jpg'
            data.append(','.join([img_name, key, class_name, str(tag_index)]) + '\n')
    with open(f'siamese_tags.csv', 'w') as f:
        f.writelines(data)

def reduce_intervals(acc, next):
    if len(acc):
        (_, end, interval, _) = acc[-1]
        if next == interval:
            acc[-1][1] += 1
            return acc
        else:
            acc.append([end + 1, end + 1, next, ""])
            return acc
    else:
        return [[1, 2, next, '']]


    
def stitch_frames_in_video(csv_file):
    frames_by_video = defaultdict(list)
    with open(csv_file, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader, None)
        for path, tag, video, interval, video_index in reader:
            frames_by_video[video].append((video_index, interval, tag, path))
    train_labels = []
    test_labels = []
    all_labels = []
    for video, data in frames_by_video.items():
        sorted_data = sorted(data, key=lambda x: int(x[0]))
        frames = list(map(lambda x: asarray(Image.open(f'curated/{x[3]}')), sorted_data))
        labels = list(map(lambda x: x[1], sorted_data))[:len(frames) -1]
        resumed_labels = reduce(reduce_intervals, labels, [])
        print(resumed_labels)
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_videofile(video)
        all_labels.append((video, resumed_labels))
        if data[0][2] == 'train':
            train_labels.append((video, resumed_labels))
        else:
            test_labels.append((video, resumed_labels))
    with open('curated_all.json', 'w') as f:
        json.dump(dict(all_labels), f)
    with open('curated_train.json', 'w') as f:
        json.dump(dict(train_labels), f)
    with open('curated_test.json', 'w') as f:
        json.dump(dict(test_labels), f)
    return dict(train_labels), dict(test_labels)

stitch_frames_in_video('curated/labels.csv')