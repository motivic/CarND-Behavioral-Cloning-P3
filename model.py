# -*- coding: utf-8 -*-

from itertools import chain
import numpy as np
import multiprocessing
import os
import pandas as pd
import pathlib
from PIL import Image
import timeit

def parse_log_entry(log_entry):
    """ Return the image frames in the log as a list of numpy arrays
    along with the steering angle

    Args:
        log_entry: One entry in the driving_log, containing paths of
            the frames from three perspectives, the steering angle,
            the throttle, the break, and the speed.

    Returns:
        A list of the images as numpy arrays and the corresponding
        steering angles.
    """
    l = log_entry.split(',')
    frames = []
    for file in l[:3]:
        image = Image.open(file)
        frames.append(np.asarray(image))
    return frames, [np.float32(l[3])]*3

def read_images(*paths):
    """ Read image frames into a 4-dimensional numpy array.

    Args:
        *paths: An iterable of such diretories.

    Returns:
        A 4-dimensional numpy array consisting of the images.
    """
    start_time = timeit.default_timer()
    pool = multiprocessing.Pool()
    images = []
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        path = pathlib.Path(p)
        images.extend(pool.map(worker, path.iterdir()))
    print("Time elapsed for reading image files: {:.2f} seconds"\
          .format(timeit.default_timer() - start_time))
    return np.array(images)


def parse_driving_log(*paths):
    """ Return a 4-dim numpy array of the image frames and a numpy
    array of the corresponding steering angles read from the logs in `paths`.

    Args:
        *paths: A driving log file or an iterable of driving log files
            containing the steering angles.

    Returns:
        A 4-dim numpy array consisting of the images and a numpy array
        of the steering angles.
    """
    start_time = timeit.default_timer()
    pool = multiprocessing.Pool()
    images = []
    steering_angles = []
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        with open(p, 'r') as file:
            results = pool.map(parse_log_entry,
                               file.readlines())
            images.extend(r[0] for r in results)
            steering_angles.extend(r[1] for r in results)
    images = list(chain.from_iterable(images))
    steering_angles = list(chain.from_iterable(steering_angles))
    print("Time elapsed for parsing driving logs: {:.2f} seconds" \
          .format(timeit.default_timer() - start_time))
    return np.array(images), np.array(steering_angles)


if __name__ == '__main__':
    HOME = os.getenv('HOME')
    LOG1 = HOME + '/data/driving_sim_track1_1/driving_log.csv'
    images, steering_angles = parse_driving_log(LOG1)
