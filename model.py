# -*- coding: utf-8 -*-

from itertools import chain
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Activation
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import os
import pathlib
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import timeit

flags = tf.app.flags
FLAGS = flags.FLAGS

# Command line flags
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('epochs', 10, 'Number of epochs')

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

def parse_driving_log(*paths):
    """ Return a list of the image frames and a list of the
    corresponding steering angles read from the logs in `paths`.

    Args:
        *paths: A driving log file or an iterable of driving log files
            containing the steering angles.

    Returns:
        A list consisting of the images and a list of the steering angles.
    """
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
    return images, steering_angles


def flip_images(images):
    """

    Args:
        images:

    Returns:

    """
    pass


def trim_imgaes(images):
    """

    Args:
        images:

    Returns:

    """
    pass


def main(_):
    # Read in images (X) and steering angles (y)
    images = []
    steering_angles = []
    HOME = os.getenv('HOME')
    path = pathlib.Path(HOME + '/data')
    for folder in path.iterdir():
        start_time = timeit.default_timer()
        if not folder.match(r'driving*'):
            continue
        file = folder.joinpath('driving_log.csv').as_posix()
        imgs, angles = parse_driving_log(file)
        print("Time elapsed for parsing {log}: {elapse:.2f} seconds" \
              .format(log=file, elapse=timeit.default_timer() - start_time))
        images.extend(imgs)
        steering_angles.extend(angles)
    X = np.array(images)
    y = np.array(steering_angles)

    # Preprocessing
    X_flipped = X[:,:,::-1,:]
    img = Image.fromarray(X_flipped[0])
    img.save('flipped.jpg')
    img = Image.fromarray(X[0])
    img.save('original.jpg')
    X_train, X_val, y_train, y_val = train_test_split(X, y)

    # Train ConvNet model


if __name__ == '__main__':
    tf.app.run()
