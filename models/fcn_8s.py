from tensorflow.keras.applications.vgg16 import VGG16

from models.utils import *


def backbone(inputs):
    """
    The backbone FCN-8s PASCAL used is ILSVRC-trained VGG16. Here we just
    take advantage of the built-in model pre-trained on ImageNet.
    """
    outputs = []

    model = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)

    for layer in ["block3_pool", "block4_pool"]:
        outputs.append(model.get_layer(layer).output)

    x = model.get_layer("block5_pool").output

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)

    outputs.append(x)
    return outputs


def fcn_8s(shape=(256, 256, 3), num_classes=1):
    """
    Fully Convolutional Networks for Semantic Segmentation.
    The `at-once` FCN-8s is fine-tuned from VGG-16 all-at-once by scaling the
    skip connections to better condition optimization.
    Paper: https://arxiv.org/pdf/1411.4038.pdf & https://arxiv.org/pdf/1605.06211.pdf
    Code from: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s-atonce/net.py
    """
    inputs = Input(shape)
    pools = backbone(inputs)

    x1 = Conv2D(21, (1, 1), padding='same')(pools[2])
    x1 = Conv2DTranspose(21, (4, 4), strides=(2, 2), padding='same')(x1)

    x2 = Conv2D(21, (1, 1), padding='same')(pools[1])
    x2 = Add()([x1, x2])
    x2 = Conv2DTranspose(21, (4, 4), strides=(2, 2), padding='same')(x2)

    x3 = Conv2D(21, (1, 1), padding='same')(pools[0])
    x3 = Add()([x2, x3])
    x3 = Conv2DTranspose(21, (16, 16), strides=(8, 8), padding='same')(x3)

    outputs = Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(x3)

    model = Model(inputs, outputs)
    return model


if __name__ == "__main__":
    model = fcn_8s()
    model.summary()
