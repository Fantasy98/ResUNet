import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

smooth = K.epsilon()


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)

    return (2. * intersection + smooth) / (union + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    alpha = 0.25
    return alpha * binary_crossentropy(y_true, y_pred) + (1 - alpha) * dice_loss(y_true, y_pred)


def weighted_cross_entropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, smooth, 1 - smooth)
    logits = tf.math.log(y_pred / (1 - y_pred))
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=y_true, pos_weight=2)

    return K.mean(loss)
