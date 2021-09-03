from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


def squeeze_excite_block(inputs, ratio=8):
    """
    Squeeze-and-Excitation block, which is one of `channel-attention` mechanisms.
    See: https://arxiv.org/pdf/1709.01507.pdf
    """
    filters = inputs.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(inputs)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    outputs = Multiply()([inputs, se])
    return outputs


def stem_block(inputs, filters, strides):
    # Conv
    x = Conv2D(filters, (3, 3), padding="same", strides=strides)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same")(x)

    # Shortcut
    s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
    s = BatchNormalization()(s)

    # Add
    outputs = Add()([x, s])
    outputs = cbam_block(outputs)
    return outputs


def aspp_block(inputs, filters):
    """
    Atrous spatial pyramid pooling to capture multi-scale context.
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
    Attention mechanism taking advantage of low-level features.
    Args:
        e: output of parallel Encoder block
        d: output of previous Decoder block
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


def cbam_block(inputs):
    """
    CBAM: Convolutional Block Attention Module, which combines
    Channel Attention Module and Spatial Attention Module, focusing on
    `what` and `where` respectively. The sequential channel-spatial order
    proves to perform best.
    See: http://dx.doi.org/10.1007/978-3-030-01234-2_1
    """
    x = channel_attention_block(inputs)
    x = spatial_attention_block(x)
    return x


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

    x = Conv2D(filters=1,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
               activation=None,
               kernel_initializer='he_normal',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    outputs = Multiply()([inputs, x])
    return outputs


def build_model(shape):
    """
    Build the model with fixed input shape [N, H, W, C].
    """
    n_filters = [32, 64, 128, 256, 512]

    inputs = Input(shape)

    # Encoder
    connections = []

    # ResNet50 as backbone
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    names = ["input_1", "conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out"]
    for name in names:
        connections.append(resnet.get_layer(name).output)

    # Bridge
    b1 = aspp_block(connections[4], n_filters[4])

    # Decoder
    """ 
    Nearest-neighbor UpSampling followed by Conv2D & ReLU to dampen checkerboard artifacts.
    See: https://distill.pub/2016/deconv-checkerboard/
    """
    d1 = attention_block(connections[3], b1)
    d1 = UpSampling2D((2, 2))(d1)
    d1 = Conv2D(n_filters[4], (3, 3), padding="same")(d1)
    d1 = Activation("relu")(d1)
    d1 = Concatenate()([d1, connections[3]])
    d1 = cbam_resblock(d1, n_filters[3])

    d2 = attention_block(connections[2], d1)
    d2 = UpSampling2D((2, 2))(d2)
    d2 = Conv2D(n_filters[3], (3, 3), padding="same")(d2)
    d2 = Activation("relu")(d2)
    d2 = Concatenate()([d2, connections[2]])
    d2 = cbam_resblock(d2, n_filters[2])

    d3 = attention_block(connections[1], d2)
    d3 = UpSampling2D((2, 2))(d3)
    d3 = Conv2D(n_filters[2], (3, 3), padding="same")(d3)
    d3 = Activation("relu")(d3)
    d3 = Concatenate()([d3, connections[1]])
    d3 = cbam_resblock(d3, n_filters[1])

    d4 = attention_block(connections[0], d3)
    d4 = UpSampling2D((2, 2))(d4)
    d4 = Conv2D(n_filters[1], (3, 3), padding="same")(d4)
    d4 = Activation("relu")(d4)
    d4 = Concatenate()([d4, connections[0]])
    d4 = cbam_resblock(d4, n_filters[0])

    # Output
    outputs = cbam_block(d4)
    outputs = Conv2D(1, (1, 1), padding="same")(outputs)
    outputs = Activation("sigmoid")(outputs)

    # Model
    model = Model(inputs, outputs)
    return model


if __name__ == "__main__":
    shape = (288, 384, 3)
    model = build_model(shape)
    model.summary()
    plot_model(model, show_shapes=True)
