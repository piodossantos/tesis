from validation import VALIDATION_DATASET
from utils import load_video
import os
import numpy as np
from statistics import mean

def correct_interval_edges(interval1, interval2, video_frames, minimium_difference):
    end_edge_index =interval1[1] - 1
    start_edge_index = interval2[0] - 1
    end_edge_image = video_frames[end_edge_index] / 255
    start_edge_image = video_frames[start_edge_index] / 255
    distance = np.linalg.norm(end_edge_image-start_edge_image)
    while distance < minimium_difference:
        start_edge_index = min(len(video_frames), start_edge_index + 1)
        start_edge_image = video_frames[start_edge_index] / 255
        distance = np.linalg.norm(end_edge_image-start_edge_image)
        if distance >= minimium_difference:
            break
        end_edge_image = max(0, end_edge_index - 1)
        end_edge_image = video_frames[end_edge_index] / 255
        distance = np.linalg.norm(end_edge_image-start_edge_image)

    adjusted_interval2 = start_edge_index + 1, interval2[1], interval2[2], interval2[3],
    adjusted_interval1 = interval1[0], end_edge_index + 1, interval1[2], interval1[3],
    return start_edge_index - end_edge_index, adjusted_interval1, adjusted_interval2


def correct_video_intervals(video_list, minimium_difference):
    curated_dataset = []
    distances = []
    for video in video_list:
        video_stream = list(load_video(f'data/{video}'))
        intervals = VALIDATION_DATASET[video]
        interval_pairs = zip(intervals, intervals[1:])
        curated_intervals = []
        for interval1, interval2 in interval_pairs:
            distance, adjusted_interval_1, adjusted_interval_2 = correct_interval_edges(interval1, interval2, video_stream, minimium_difference)
            distances.append(distance)
            interval2[0], interval2[1] = adjusted_interval_2[0], adjusted_interval_2[1]
            interval1[0], interval1[1] = adjusted_interval_1[0], adjusted_interval_1[1]
            if interval1 not in curated_intervals:
                curated_intervals.append(interval1)
            curated_intervals.append(interval2)
        curated_dataset.append((video, curated_intervals))
    return dict(curated_dataset), max(distances), mean(distances)

def prepare_siamese_dataset(videos):
    if not os.path.exists('siamese'):
        os.mkdir('siamese')
    already_images = set()
    

curated_dataset, max_value, avg = correct_video_intervals(list(VALIDATION_DATASET.keys()), 2)
import json
print(json.dumps(curated_dataset))
print(max_value, avg)