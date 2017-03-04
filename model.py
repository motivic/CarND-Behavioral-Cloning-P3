# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing
import os
import pandas as pd
import pathlib
from PIL import Image

def worker(files):
    images = []
    for file in files:
        image = Image.open(file)
        images.append(np.assarray(image))

def read_images(*paths):
    """

    Args:
        *paths: A directory containing images or
            a list of such diretories.

    Returns:
        A numpy array consisting of the images.
    """
    images = []
    filenames = []
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        path = pathlib.Path(p)
        for file in path.iterdir():
            filenames.append(file)
            image = Image.open(file)
            images.append(np.asarray(image))
    return np.array(images), filenames


def read_steering_angle(*paths):
    """

    Args:
        *paths: A file containing the steering angle or
            a

    Returns:

    """

if __name__ == '__main__':
    HOME = os.getenv('HOME')
    images, filenames = read_images(HOME + '/data/driving_sim_track1_1/IMG')
