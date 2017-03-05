# -*- coding: utf-8 -*-

from itertools import chain
from keras.models import Model, Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda
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
    angle = np.float32(l[3])
    frames = []
    for file in l[:3]:
        image = Image.open(file)
        frames.append(np.asarray(image))
    return frames, [angle, angle + 0.2, angle - 0.2]


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


def train_model(X_train, y_train):
    """ Train and save a convolutional neural network based on
    the one proposed by Nvidia.

    Args:
        X_train: Features data consisting of images.
        y_train: Steering angles.
    """
    model = Sequential()
    # Cropping
    model.add(Cropping2D(cropping=((50, 20), (0, 0)),
                         input_shape=(160, 320, 3)))
    # Normalizing and zero-centering
    model.add(Lambda(lambda x: x/255 - 0.5,
                     input_shape=(90, 320, 3)))

    # Follow Nvidia's architecture
    # Convolutional layer 1
    model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2),
                     input_shape=(90, 320, 3)))
    # Convolutional layer 2
    model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2),
                     input_shape=(43, 158, 24)))
    # Convolutional layer 3
    model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2),
                     input_shape=(20, 77, 36)))
    # Convolutional layer 4
    model.add(Conv2D(64, 3, 3, activation='relu', subsample=(2, 2),
                     input_shape=(8, 37, 48)))
    # Convolutional layer 5
    model.add(Conv2D(72, 3, 3, activation='relu', subsample=(1, 1),
                     input_shape=(3, 18, 64)))
    # Fully-connected layer 1
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    # Fully-connected layer 2
    model.add(Dense(50, activation='relu'))
    # Fully-connected layer 3
    model.add(Dense(10, activation='relu'))
    # Fully-connected layer 4
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2,
              batch_size=FLAGS.batch_size, shuffle=True,
              nb_epoch=FLAGS.epochs, verbose=1)
    model.save('model.h5')


def main(_):
    """ Reads in images and trains convolutional neural network.
    """
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
        break
    X = np.array(images)
    y = np.array(steering_angles)

    # Preprocessing
    # Add horizontally flipped images
    X_flipped = X[:,:,::-1,:]
    X = np.concatenate([X, X_flipped], 0)
    y = np.concatenate([y, np.negative(y)], 0)

    # Train ConvNet model
    train_model(X, y)

if __name__ == '__main__':
    tf.app.run()
