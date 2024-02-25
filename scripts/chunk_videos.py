from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from utils import load_video
import uuid
from tapAndScroll import TRAIN_TAPS, TEST_TAPS
import cv2
from functools import partial


def is_frame_from_tap_interval(frame_number, class_label, interval):
    start, end, class_name = interval
    return (class_name.upper() == class_label) and start <= frame_number <= end

def gen_chunk_videos(dataset, csv_file, clip_len=16, frame_rate=15):
    data = ['filepath,loading,scroll,tap\n']
    for video, intervals in dataset.items():
        stream = load_video(f'data/{video}')
        chunk = []
        loading = False
        scroll = False
        tap = False
        for index, frame in enumerate(stream):
            scroll_filter = partial(is_frame_from_tap_interval, index + 1, 'SCROLL')
            loading_filter = partial(is_frame_from_tap_interval, index + 1, 'LOADING')
            tap_filter = partial(is_frame_from_tap_interval, index + 1, 'TAP')
            scrolls = list(filter(scroll_filter, intervals))
            scroll = len(scrolls) > 0 or scroll
            loadings = list(filter(loading_filter, intervals))
            loading = len(loadings) > 0 or loading
            taps = list(filter(tap_filter, intervals))
            tap = len(taps) > 0 or tap
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame.resize((128, 171), refcheck=False)
            chunk.append(frame)
            if index % clip_len == clip_len - 1:
                print(len(chunk))
                chunk_video = ImageSequenceClip(chunk, frame_rate)
                path = f'{uuid.uuid4().hex}.mp4'
                chunk_video.write_videofile(f'chunks/{path}')
                data.append(','.join([path, str(int(loading)), str(int(scroll)), str(int(tap))]) + '\n')
                chunk = []
                loading = False
                scroll = False
                tap = False
    with open(f'chunks/{csv_file}', 'w') as f:
        f.writelines(data)


gen_chunk_videos(TRAIN_TAPS, 'train.csv', 16)
gen_chunk_videos(TEST_TAPS, 'test.csv', 16)
