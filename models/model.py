from models.utils import *


def stem_block(inputs, filters, strides):
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

    # CBAM
    x = cbam_block(x)

    # Shortcut
    s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
    s = BatchNormalization()(s)

    # Add
    outputs = Add()([x, s])
    return outputs


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


def res_block_no_stride(inputs, filters):
    """
    Residual block with full pre-activation (BN-ReLU-weight-BN-ReLU-weight).
    See: https://arxiv.org/pdf/1512.03385.pdf & https://arxiv.org/pdf/1603.05027v3.pdf
    """
    # Conv
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)

    # Add
    outputs = Add()([x, inputs])
    return outputs


def aspp_block(inputs, filters):
    """
    Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context.
    See: https://arxiv.org/pdf/1706.05587.pdf
    """
    shape = inputs.shape

    x1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    x1 = Conv2D(filters, 1, padding="same")(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(x1)

    x2 = Conv2D(filters, 1, dilation_rate=1, padding="same", use_bias=False)(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x3 = Conv2D(filters, 3, dilation_rate=6, padding="same", use_bias=False)(inputs)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)

    x4 = Conv2D(filters, 3, dilation_rate=12, padding="same", use_bias=False)(inputs)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)

    x5 = Conv2D(filters, 3, dilation_rate=18, padding="same", use_bias=False)(inputs)
    x5 = BatchNormalization()(x5)
    x5 = Activation("relu")(x5)

    x = Add()([x1, x2, x3, x4, x5])

    x = Conv2D(filters, 1, dilation_rate=1, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    outputs = Activation("relu")(x)

    return outputs


def attention_block(e, d):
    """
    Attention block taking advantage of low-level features.
    Args:
        e: output of parallel Encoder block
        d: output of previous Decoder block
    See: https://arxiv.org/pdf/1804.03999.pdf
    """
    filters = d.shape[-1]

    x1 = BatchNormalization()(e)
    x1 = Activation("relu")(x1)
    x1 = Conv2D(filters, (3, 3), padding="same")(x1)

    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)

    x2 = BatchNormalization()(d)
    x2 = Activation("relu")(x2)
    x2 = Conv2D(filters, (3, 3), padding="same")(x2)

    x = Add()([x1, x2])

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same")(x)

    outputs = Multiply()([x, d])
    return outputs


def channel_attention_block(inputs, ratio=16):
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


def spatial_attention_block(inputs):
    """
    Spatial Attention Module utilizing the inter-spatial relationship of features.
    """
    kernel_size = 7

    # avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(inputs)
    # max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(inputs)
    avg_pool = K.mean(inputs, axis=-1, keepdims=True)
    max_pool = K.max(inputs, axis=-1, keepdims=True)

    x = Concatenate()([avg_pool, max_pool])

    x = Conv2D(1, kernel_size, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    outputs = Multiply()([inputs, x])
    return outputs


def cbam_block(inputs):
    """
    CBAM: Convolutional Block Attention Module, which combines
    Channel Attention Module and Spatial Attention Module, focusing on
    `what` and `where` respectively. The sequential channel-spatial order
    proves to perform best.
    See: https://arxiv.org/pdf/1807.06521.pdf
    """
    x = channel_attention_block(inputs)
    x = spatial_attention_block(x)
    return x


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


def cbam_resblock_no_stride(inputs, filters):
    """
    Residual block with Convolutional Block Attention Module.
    """
    # Conv
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)

    # CBAM
    x = cbam_block(x)

    # Add
    outputs = Concatenate()([x, inputs])

    outputs = Conv2D(filters, (3, 3), padding="same")(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation('relu')(outputs)

    return outputs


def build_model(shape=(256, 256, 3), num_classes=1):
    """
    Build a model with fixed input shape [N, H, W, C].
    """
    n_filters = [32, 64, 128, 256, 512]

    inputs = Input(shape)

    # Encoder
    c0 = cbam_block(inputs)

    c1 = cbam_resblock_no_stride(c0, n_filters[0])
    c1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c1)

    c2 = cbam_resblock_no_stride(c1, n_filters[1])
    c2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c2)

    c3 = cbam_resblock_no_stride(c2, n_filters[2])
    c3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c3)

    # Bridge
    # b1 = aspp_block(c3, n_filters[3])
    b1 = cbam_resblock_no_stride(c3, n_filters[3])
    # b1 = cbam_block(c3)

    # Decoder
    """ 
    Nearest-neighbor UpSampling followed by Conv2D & ReLU to dampen checkerboard artifacts.
    See: https://distill.pub/2016/deconv-checkerboard/
    """
    d1 = attention_block(c2, b1)
    d1 = Conv2D(n_filters[2], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(d1))
    d1 = Concatenate()([d1, c2])
    d1 = cbam_resblock_no_stride(d1, n_filters[2])

    d2 = attention_block(c1, d1)
    d2 = Conv2D(n_filters[1], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(d2))
    d2 = Concatenate()([d2, c1])
    d2 = cbam_resblock_no_stride(d2, n_filters[1])

    d3 = attention_block(c0, d2)
    d3 = Conv2D(n_filters[0], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(d3))
    d3 = Concatenate()([d3, c0])
    d3 = cbam_resblock_no_stride(d3, n_filters[0])

    # Output
    outputs = Conv2D(num_classes, (1, 1), padding="same")(d3)
    outputs = Activation("sigmoid")(outputs)

    # Model
    model = Model(inputs, outputs)
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
