from collections import defaultdict
import utils

def get_frames_by_cluster(path, clusters):
    frames_by_cluster = defaultdict(list)
    stream = list(utils.load_video(path))
    for index, frame in enumerate(stream):
        label = clusters[index]
        frames_by_cluster[label].append(frame)
    return frames_by_cluster