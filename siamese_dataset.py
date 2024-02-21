from validation import VALIDATION_DATASET
from utils import load_video
import hashlib
import cv2
from PIL import Image


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

get_labels_by_hash(VALIDATION_DATASET)