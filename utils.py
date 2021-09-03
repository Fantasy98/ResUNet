import os
import re
from glob import glob

import cv2
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

from metrics import *


def create_dir(path):
    """
    Create directories if not exist.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Creating directory {path} failed")


def read_data(image, mask):
    """
    Read the image and mask in COLOR mode.
    """
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask, cv2.IMREAD_COLOR)

    return image, mask


def read_image(image):
    """
    Read and normalize the image. Resize it for cross-dataset evaluation.
    """
    if isinstance(image, bytes):
        image = image.decode()
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (384, 288))
    image = np.clip(image - np.median(image) + 127, 0, 255)
    image = image / 255.
    image = image.astype(np.float32)

    return image


def read_mask(mask):
    """
    Read and normalize the mask. Resize it for cross-dataset evaluation.
    """
    if isinstance(mask, bytes):
        mask = mask.decode()
    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (384, 288))
    mask = mask / 255.
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)

    return mask


def parse_data(images, masks):
    """
    Cast input data type to tf.float32.
    """

    def _parse(image, mask):
        image = read_image(image)
        mask = read_mask(mask)
        return image, mask

    images, masks = tf.numpy_function(_parse, [images, masks], [tf.float32, tf.float32])
    images.set_shape([288, 384, 3])
    masks.set_shape([288, 384, 1])

    return images, masks


def tf_dataset(images, masks, batch_size=8):
    """
    Generate tf dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=32)
    dataset = dataset.map(map_func=parse_data)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    return dataset


def shuffle_data(images, masks):
    """
    Shuffle data in a consistent way.
    """
    images, masks = shuffle(images, masks, random_state=42)
    return images, masks


def load_model_weight(path):
    """
    Load the trained model.
    """
    with CustomObjectScope({
        'dice_coef': dice_coef,
        'dice_loss': dice_loss,
        'focal_loss': focal_loss,
        'dice_focal_loss': dice_focal_loss,
        'dice_topk_loss': dice_topk_loss,
        'focal_tversky_loss': focal_tversky_loss
    }):
        model = load_model(path)

    return model


def load_dataset(path, cross_dataset=True):
    """
    Load dataset from the given path.
    Use regex to distinguish the path due to their different hierarchies.
    """
    if cross_dataset:
        if re.search("Clinic", path, re.IGNORECASE):
            images = sorted(glob(os.path.join(path, "Original/*")))
            masks = sorted(glob(os.path.join(path, "Ground Truth/*")))
        elif re.search("ETIS", path, re.IGNORECASE):
            images = sorted(glob(os.path.join(path, "ETIS-LaribPolypDB/*")))
            masks = sorted(glob(os.path.join(path, "Ground Truth/*")))
        else:
            images = sorted(glob(os.path.join(path, "images/*")))
            masks = sorted(glob(os.path.join(path, "masks/*")))
    else:
        images = sorted(glob(os.path.join(path, "images/*")))
        masks = sorted(glob(os.path.join(path, "masks/*")))

    return images, masks
