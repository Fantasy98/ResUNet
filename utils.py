import os

import cv2
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

from metrics import *


def create_dir(path):
    """
    Create a directory if not exist.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")


def read_data(x, y):
    """
    Read the image and mask from the given path.
    """
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    mask = cv2.imread(y, cv2.IMREAD_COLOR)
    return image, mask


def read_image(x):
    x = x.decode()
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = np.clip(image - np.median(image) + 127, 0, 255)
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def read_mask(y):
    y = y.decode()
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = mask / 255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def parse_data(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([288, 384, 3])
    y.set_shape([288, 384, 1])
    return x, y


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=32)
    dataset = dataset.map(map_func=parse_data)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch)
    return dataset


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def load_model_weight(path):
    with CustomObjectScope({
        'dice_coef': dice_coef,
        'dice_loss': dice_loss,
        'bce_dice_loss': bce_dice_loss,
        'weighted_cross_entropy': weighted_cross_entropy
    }):
        model = load_model(path)
    return model
