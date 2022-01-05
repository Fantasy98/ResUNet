from models.utils import *


def ca_stem_block(inputs, filters, strides=1):
    """
    Residual block for the first layer of Deep Residual U-Net.
    See: https://arxiv.org/pdf/1711.10684.pdf
    Code from: https://github.com/dmolony3/ResUNet
    """
    # Conv
    x = Conv2D(filters, (3, 3), padding="same", strides=strides)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same")(x)

    # CA
    x = ca_block(x)

    # Shortcut
    s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
    s = BatchNormalization()(s)

    # Add
    outputs = Add()([x, s])
    return outputs


def feature_fusion(high, low):
    """
    Low- and high-level feature fusion, taking advantage of low-level contextual information.

    Args:
        high: high-level semantic information.
        low: low-level feature map with more contextual information.

    See: https://arxiv.org/pdf/1804.03999.pdf
    """
    filters = low.shape[-1]

    x1 = UpSampling2D(size=(2, 2))(high)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = Conv2D(filters, (3, 3), padding="same")(x1)

    x2 = BatchNormalization()(low)
    x2 = Activation("relu")(x2)
    x2 = Conv2D(filters, (3, 3), padding="same")(x2)

    x = Add()([x1, x2])

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same")(x)

    outputs = Multiply()([x, low])
    return outputs


def ca_block(inputs, ratio=16):
    """
    Channel Attention Module exploiting the inter-channel relationship of features.
    """
    shape = inputs.shape
    filters = shape[-1]

    # avg_pool = Lambda(lambda x: K.mean(x, axis=[1, 2], keepdims=True))(inputs)
    # max_pool = Lambda(lambda x: K.max(x, axis=[1, 2], keepdims=True))(inputs)
    # avg_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    # max_pool = MaxPooling2D(pool_size=(shape[1], shape[2]))(inputs)
    avg_pool = K.mean(inputs, axis=[1, 2], keepdims=True)
    max_pool = K.max(inputs, axis=[1, 2], keepdims=True)

    x1 = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(avg_pool)
    x1 = Dense(filters, activation=None, kernel_initializer='he_normal', use_bias=False)(x1)

    x2 = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(max_pool)
    x2 = Dense(filters, activation=None, kernel_initializer='he_normal', use_bias=False)(x2)

    x = Add()([x1, x2])
    x = Activation('sigmoid')(x)

    outputs = Multiply()([inputs, x])
    return outputs


def sa_block(inputs):
    """
    Spatial Attention Module utilizing the inter-spatial relationship of features.
    """
    kernel_size = 7

    # avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(inputs)
    # max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(inputs)
    avg_pool = K.mean(inputs, axis=-1, keepdims=True)
    max_pool = K.max(inputs, axis=-1, keepdims=True)

    x = Concatenate()([avg_pool, max_pool])

    x = Conv2D(1, kernel_size, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)

    outputs = Multiply()([inputs, x])
    return outputs


def cbam_block(inputs):
    """
    CBAM: Convolutional Block Attention Module, which combines Channel Attention Module and Spatial Attention Module,
    focusing on `what` and `where` respectively. The sequential channel-spatial order proves to perform best.
    See: https://arxiv.org/pdf/1807.06521.pdf
    """
    x = ca_block(inputs)
    x = sa_block(x)
    return x


def res_block(inputs, filters, strides=1):
    """
    Residual block with full pre-activation (BN-ReLU-weight-BN-ReLU-weight).
    See: https://arxiv.org/pdf/1512.03385.pdf & https://arxiv.org/pdf/1603.05027v3.pdf
    """
    # Conv
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=strides)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)

    # Shortcut
    s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
    s = BatchNormalization()(s)

    # Add
    outputs = Add()([x, s])
    return outputs


def ca_resblock(inputs, filters, strides=1):
    """
    Residual block with Channel Attention Module.
    """
    # Conv
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=strides)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)

    # CA
    x = ca_block(x)

    # Shortcut
    s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
    s = BatchNormalization()(s)

    # Add
    outputs = Add()([x, s])
    return outputs


def sa_resblock(inputs, filters, strides=1):
    """
    Residual block with Spatial Attention Module.
    """
    # Conv
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=strides)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)

    # SA
    x = sa_block(x)

    # Shortcut
    s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
    s = BatchNormalization()(s)

    # Add
    outputs = Add()([x, s])
    return outputs


def cbam_resblock(inputs, filters, strides=1):
    """
    Residual block with Convolutional Block Attention Module.
    """
    # Conv
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=strides)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)

    # CBAM
    x = cbam_block(x)

    # Shortcut
    s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
    s = BatchNormalization()(s)

    # Add
    outputs = Add()([x, s])
    return outputs


def build_model(shape=(256, 256, 3), num_classes=1):
    """
    Build a model with fixed input shape [N, H, W, C].
    """
    n_filters = [32, 64, 128, 256, 512]

    inputs = Input(shape)

    # Encoder
    c0 = ca_stem_block(inputs, n_filters[0])
    c1 = cbam_resblock(c0, n_filters[1], strides=2)
    c2 = cbam_resblock(c1, n_filters[2], strides=2)
    c3 = cbam_resblock(c2, n_filters[3], strides=2)

    # Bridge
    b1 = sa_resblock(c3, n_filters[4])

    # Decoder
    # Nearest-neighbor UpSampling followed by Conv2D & ReLU to dampen checkerboard artifacts.
    # See: https://distill.pub/2016/deconv-checkerboard/

    d1 = UpSampling2D(size=(2, 2))(b1)
    d1 = Conv2D(n_filters[3], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(d1)
    d1 = feature_fusion(c3, d1)
    d1 = Concatenate()([d1, c2])
    d1 = cbam_resblock(d1, n_filters[3])

    d2 = UpSampling2D(size=(2, 2))(d1)
    d2 = Conv2D(n_filters[2], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(d2)
    d2 = feature_fusion(c2, d2)
    d2 = Concatenate()([d2, c1])
    d2 = cbam_resblock(d2, n_filters[2])

    d3 = UpSampling2D(size=(2, 2))(d2)
    d3 = Conv2D(n_filters[1], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(d3)
    d3 = feature_fusion(c1, d3)
    d3 = Concatenate()([d3, c0])
    d3 = cbam_resblock(d3, n_filters[1])

    # Output
    outputs = ca_resblock(d3, n_filters[0])
    outputs = Conv2D(num_classes, (1, 1), padding="same")(outputs)
    outputs = Activation("sigmoid")(outputs)

    # Model
    model = Model(inputs, outputs)
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
