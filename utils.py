import os
import re
from glob import glob

import cv2
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tifffile import tifffile

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


def get_filename(file):
    """
    Extract filename.
    """
    filename = file.split("\\")[-1].split(".")[0]
    return filename


def get_extension(file):
    """
    Extract extension.
    """
    extension = file.split("\\")[-1].split(".")[-1]
    return extension


def read_data(image, mask):
    """
    Read the image and mask with shape [H, W, C] and [H, W] respectively.
    Use `tifffile` to read .tiff images instead of `cv2`, because the latter has trouble
    dealing with CVC-ClinicDB original images (colorful images become, er, grayscale?).
    """
    if isinstance(image, bytes):
        image = image.decode()

    if isinstance(mask, bytes):
        mask = mask.decode()

    image_ext = get_extension(image)
    mask_ext = get_extension(mask)

    if image_ext in ["tif", "tiff"]:
        image = tifffile.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # cv2 uses BGR channel order
    else:
        image = cv2.imread(image, cv2.IMREAD_COLOR)

    if mask_ext in ["tif", "tiff"]:
        mask = tifffile.imread(mask)
    else:
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    return image, mask


def normalize(image):
    """
    Normalize the image/mask. Resize it for cross-dataset evaluation.
    """
    image = cv2.resize(image, (384, 288))
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


def parse_data(images, masks):
    """
    Cast input data type to tf.float32.
    """

    def _parse(image, mask):
        image, mask = read_and_normalize_data(image, mask)
        mask = np.expand_dims(mask, axis=-1)
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
