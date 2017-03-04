# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing
import os
import pandas as pd
import pathlib
from PIL import Image
import timeit

def worker(file):
    image = Image.open(file)
    return np.asarray(image)

def read_images(*paths):
    """

    Args:
        *paths: A directory containing images or
            a list of such diretories.

    Returns:
        A numpy array consisting of the images.
    """
    start_time = timeit.default_timer()
    pool = multiprocessing.Pool()
    images = []
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        path = pathlib.Path(p)
        #images.extend(pool.map(worker, path.iterdir()))

        for file in path.iterdir():
            image = Image.open(file)
            images.append(np.asarray(image))

    print("Time elapsed for reading image files: {:.2f} seconds"\
          .format(timeit.default_timer() - start_time))
    return np.array(images)


def read_steering_angle(*paths):
    """

    Args:
        *paths: A file containing the steering angle or
            a

    Returns:

    """

if __name__ == '__main__':
    HOME = os.getenv('HOME')
    images = read_images(HOME + '/data/driving_sim_track1_1/IMG')
    print(images)
