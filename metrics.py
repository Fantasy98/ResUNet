import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

"""
Some loss functions from `Loss functions for image segmentation`.
Paper: https://doi.org/10.1016/j.media.2021.102035
See: https://github.com/JunMa11/SegLoss & https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py
"""

smooth = K.epsilon()


def flatten(y_true, y_pred):
    """
    Flatten tensors to 1D.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    return y_true, y_pred


def convert_to_logits(y_pred):
    """
    Convert a tensor to logits.
    """
    y_pred = K.clip(y_pred, smooth, 1 - smooth)
    logits = K.log(y_pred / (1 - y_pred))

    return logits


def cross_entropy(y_true, y_pred):
    """
    Sigmoid cross entropy.
    """
    logits = convert_to_logits(y_pred)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_true)

    return K.mean(loss)


def weighted_cross_entropy(y_true, y_pred):
    """
    Weighted cross entropy for skewed data.
    See: https://arxiv.org/pdf/1505.04597.pdf
    """
    beta = 0.25
    pos_weight = beta / (1 - beta)

    logits = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=y_true, pos_weight=pos_weight)

    return K.mean(loss)


def dice_coef(y_true, y_pred):
    """
    Sørensen–Dice coefficient (aka F1 score).
    See: https://arxiv.org/pdf/1608.04117.pdf
    """
    y_true, y_pred = flatten(y_true, y_pred)

    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)

    return (2. * intersection + smooth) / (union + smooth)


def dice_loss(y_true, y_pred):
    loss = 1. - dice_coef(y_true, y_pred)
    return loss


def dice_coef_squared(y_true, y_pred):
    """
    Dice coefficient with square.
    Square the terms in the denominator as proposed by Milletari et al.
    See: https://arxiv.org/pdf/1606.04797.pdf
    """
    y_true, y_pred = flatten(y_true, y_pred)

    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true ** 2) + K.sum(y_pred ** 2)

    return (2. * intersection + smooth) / (union + smooth)


def dice_loss_squared(y_true, y_pred):
    loss = 1. - dice_coef_squared(y_true, y_pred)
    return loss


def generalized_dice_coef(y_true, y_pred):
    """
    Generalised dice loss for imbalanced data.
    See: https://arxiv.org/pdf/1707.03237.pdf
    Code from https://github.com/junqiangchen/Image-Segmentation-Loss-Functions/blob/master/loss_function.py#L121
    """
    w = K.sum(y_true, axis=[1, 2])
    w = 1. / (w ** 2 + smooth)

    numerator = w * K.sum(y_true * y_pred, axis=[1, 2])
    numerator = K.sum(numerator, axis=1)

    denominator = w * K.sum(y_true + y_pred, axis=[1, 2])
    denominator = K.sum(denominator, axis=1)

    return (2. * numerator + smooth) / (denominator + smooth)


def generalized_dice_loss(y_true, y_pred):
    loss = 1. - K.mean(generalized_dice_coef(y_true, y_pred))
    return loss


def ce_dice_loss(y_true, y_pred):
    """
    A combination of cross entropy and dice loss.
    See: https://arxiv.org/pdf/1809.10486.pdf
    """
    loss = cross_entropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss / 2.


def combo_loss(y_true, y_pred):
    """
    A weighted combination of cross entropy and dice loss.
    See: https://arxiv.org/pdf/1805.02798.pdf
    Code from https://github.com/asgsaeid/ComboLoss/blob/master/combo_loss.py
    """
    alpha = 0.5
    loss = alpha * cross_entropy(y_true, y_pred) + (1 - alpha) * dice_loss(y_true, y_pred)

    return loss


def focal_loss(y_true, y_pred):
    """
    Focal loss focusing more on hard examples.
    See: https://arxiv.org/pdf/1708.02002.pdf
    Code from https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py#L54
    """
    alpha = 0.25
    gamma = 2.

    def focal_loss_with_logits(logits, labels, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * labels
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - labels)

        return (tf.math.log1p(K.exp(-K.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    logits = convert_to_logits(y_pred)
    loss = focal_loss_with_logits(logits=logits, labels=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

    return K.mean(loss)


def dice_focal_loss(y_true, y_pred):
    """
    A combination of dice and focal loss.
    See: https://arxiv.org/pdf/2102.10446v1.pdf
    """
    loss = dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)
    return loss / 2.


def topk_loss(y_true, y_pred):
    """
    TopK loss of K most hardest pixels. Proved to perform best when k = 512.
    See: https://arxiv.org/pdf/1605.06885.pdf
    Code from https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/ND_Crossentropy.py#L34
    """
    ce = cross_entropy(y_true, y_pred)
    values, indices = tf.nn.top_k(ce, k=512)

    return K.mean(values)


def dice_topk_loss(y_true, y_pred):
    """
    A combination of dice and topK loss.
    See: https://arxiv.org/pdf/2101.00232.pdf
    Code from https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py#L462
    """
    loss = dice_loss(y_true, y_pred) + topk_loss(y_true, y_pred)
    return loss / 2.


def tversky(y_true, y_pred):
    """
    Tversky loss for imbalanced data.
    See: https://arxiv.org/pdf/1706.05721.pdf
    """
    alpha = 0.3
    y_true, y_pred = flatten(y_true, y_pred)

    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))

    return (tp + smooth) / (tp + alpha * fp + (1 - alpha) * fn + smooth)


def tversky_loss(y_true, y_pred):
    return 1. - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred):
    """
    A combination of focal and Tversky loss.
    See: https://arxiv.org/pdf/1810.07842.pdf
    Code from https://github.com/nabsabraham/focal-tversky-unet/blob/347d39117c24540400dfe80d106d2fb06d2b99e1/losses.py#L65
    """
    gamma = 0.75
    loss = tversky_loss(y_true, y_pred)

    return K.pow(loss, gamma)


def log_cosh_dice_loss(y_true, y_pred):
    """
    Log cosh dice loss for skewed data.
    See: https://arxiv.org/pdf/2006.14822.pdf
    Code from https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py#L135
    """
    x = dice_loss(y_true, y_pred)
    return K.log((K.exp(x) + K.exp(-x)) / 2.)


def exp_log_loss(y_true, y_pred):
    """
    Exponential logarithmic loss for imbalanced data.
    See: https://arxiv.org/pdf/1809.00076.pdf
    Code from https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py#L135
    """
    gamma = 0.3
    dice = dice_loss(y_true, y_pred)
    wce = weighted_cross_entropy(y_true, y_pred)
    loss = 0.8 * K.pow(-K.log(K.clip(dice, smooth, 1 - smooth)), gamma) + 0.2 * wce

    return loss
