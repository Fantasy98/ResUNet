import os
import re
from glob import glob

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tifffile import tifffile

from losses import *


def create_dirs(path):
    """
    Create directories if not exist.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Creating directory {path} failed")


def get_filename(file):
    """
    Extract filename.
    Be careful with the file separator: use "\\" if "/" doesn't work.
    """
    filename = file.split("/")[-1].split(".")[0]
    return filename


def get_extension(file):
    """
    Extract extension.
    """
    extension = file.split("/")[-1].split(".")[-1]
    return extension


def read_data(image, mask):
    """
    Read the image and mask with shape [H, W, C] and [H, W] respectively.
    Use `tifffile` to read TIFF images instead of `cv2`, because the latter has trouble
    dealing with CVC-ClinicDB original images. (Colorful images become, er, grayscale?)
    """
    if isinstance(image, bytes):
        image = image.decode()

    if isinstance(mask, bytes):
        mask = mask.decode()

    extensions = ["tif", "tiff"]

    image_ext = get_extension(image)
    mask_ext = get_extension(mask)

    # > For historical reasons, OpenCV reads an image in BGR format. Albumentations uses
    # the most common and popular RGB image format. So when using OpenCV, we need to
    # convert the image format to RGB explicitly.
    # But it is fine to perform all these operations in BGR color space.
    if image_ext in extensions:
        image = tifffile.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR color space
    else:
        image = cv2.imread(image, cv2.IMREAD_COLOR)

    if mask_ext in extensions:
        mask = tifffile.imread(mask)
    else:
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    return image, mask


def normalize(image):
    """
    Normalize the image/mask, and resize it to a fixed shape of (256, 256).
    """
    image = cv2.resize(image, (256, 256))
    image = image / 255.
    image = image.astype(np.float32)

    return image


def read_and_normalize_data(image, mask):
    """
    Read and normalize the image and mask.
    """
    image, mask = read_data(image, mask)
    image = normalize(image)
    mask = normalize(mask)

    return image, mask


def map_func(images, masks):
    """
    Read and normalize images and masks, and cast data type to tf.float32.
    """

    def func(image, mask):
        image, mask = read_and_normalize_data(image, mask)
        mask = np.expand_dims(mask, axis=-1)

        return image, mask

    images, masks = tf.numpy_function(func=func, inp=[images, masks], Tout=[tf.float32, tf.float32])

    # Restore dataset shapes (the dataset loses its shape after applying a tf.numpy_function)
    images.set_shape([256, 256, 3])
    masks.set_shape([256, 256, 1])

    return images, masks


def tf_dataset(images, masks, batch_size=8):
    """
    Generate tf dataset. There is a difference in the order of `shuffle`, `repeat` and `batch`.
    `Repeat` before `shuffle` seems to provide better performance.
    See: https://stackoverflow.com/questions/49915925/output-differences-when-changing-order-of-batch-shuffle-and-repeat
    """
    dataset = tf.data.Dataset.from_tensor_slices(tensors=(images, masks)) \
        .repeat() \
        .shuffle(buffer_size=len(images)) \
        .map(map_func=map_func) \
        .batch(batch_size=batch_size) \
        .prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def load_dataset(path, cross_dataset=True):
    """
    Load dataset from the given path.
    Use regex to distinguish the paths due to their different hierarchies.
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


def load_model_weights(path):
    """
    Load the trained model.
    """
    with CustomObjectScope({
        'dice_coef': dice_coef,
        'dice_loss': dice_loss,
        'focal_loss': focal_loss
    }):
        model = load_model(path)

    return model
