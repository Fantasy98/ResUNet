from models.utils import *


def res_block(inputs, filters, strides=(1, 1)):
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
    x = Conv2D(filters, (3, 3), padding="same", strides=(1, 1))(x)

    # Shortcut
    s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
    s = BatchNormalization()(s)

    # Add
    outputs = Add()([x, s])
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
    kernel_size = (7, 7)

    # avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(inputs)
    # max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(inputs)
    avg_pool = K.mean(inputs, axis=-1, keepdims=True)
    max_pool = K.max(inputs, axis=-1, keepdims=True)

    x = Concatenate()([avg_pool, max_pool])

    x = Conv2D(filters=1, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
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


def cbam_resblock(inputs, filters, strides=(1, 1)):
    """
    Residual block with Convolutional Block Attention Module.
    """
    # Conv
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=strides)(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", strides=(1, 1))(x)

    # CBAM
    x = cbam_block(x)

    # Shortcut
    s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
    s = BatchNormalization()(s)

    # Add
    outputs = Add()([x, s])
    return outputs


def build_model(shape, num_class=1, deep_supervision=False):
    """
    Residual UNet++.
    Based on UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation.

    All pooling operations are replaced with convolutional layers with strides of 2 because the latter can learn
    all necessary invariances, which is especially the case when the network is large enough.
    See: https://arxiv.org/pdf/1412.6806.pdf

    We use nearest-neighbor UpSampling followed by Conv2D & ReLU to dampen checkerboard artifacts.
    See: https://distill.pub/2016/deconv-checkerboard/
    """
    n_filters = [32, 64, 128, 256, 512]

    inputs = Input(shape=shape, name='main_input')

    conv1_1 = cbam_resblock(inputs, filters=n_filters[0])

    conv2_1 = cbam_resblock(conv1_1, filters=n_filters[1], strides=(2, 2))

    up1_2 = Conv2D(n_filters[0], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), name='up12')(conv2_1))

    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=3)
    conv1_2 = cbam_resblock(conv1_2, filters=n_filters[0])

    conv3_1 = cbam_resblock(conv2_1, filters=n_filters[2], strides=(2, 2))

    up2_2 = Conv2D(n_filters[1], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), name='up22')(conv3_1))

    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=3)
    conv2_2 = cbam_resblock(conv2_2, filters=n_filters[1])

    up1_3 = Conv2D(n_filters[0], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), name='up13')(conv2_2))

    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=3)
    conv1_3 = cbam_resblock(conv1_3, filters=n_filters[0])

    conv4_1 = cbam_resblock(conv3_1, filters=n_filters[3], strides=(2, 2))

    up3_2 = Conv2D(n_filters[2], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), name='up32')(conv4_1))

    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=3)
    conv3_2 = cbam_resblock(conv3_2, filters=n_filters[2])

    up2_3 = Conv2D(n_filters[1], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), name='up23')(conv3_2))

    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=3)
    conv2_3 = cbam_resblock(conv2_3, filters=n_filters[1])

    up1_4 = Conv2D(n_filters[0], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), name='up14')(conv2_3))

    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=3)
    conv1_4 = cbam_resblock(conv1_4, filters=n_filters[0])

    conv5_1 = cbam_resblock(conv4_1, filters=n_filters[4], strides=(2, 2))

    up4_2 = Conv2D(n_filters[3], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), name='up42')(conv5_1))

    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=3)
    conv4_2 = cbam_resblock(conv4_2, filters=n_filters[3])

    up3_3 = Conv2D(n_filters[2], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), name='up33')(conv4_2))

    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=3)
    conv3_3 = cbam_resblock(conv3_3, filters=n_filters[2])

    up2_4 = Conv2D(n_filters[1], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), name='up24')(conv3_3))

    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=3)
    conv2_4 = cbam_resblock(conv2_4, filters=n_filters[1])

    up1_5 = Conv2D(n_filters[0], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2), name='up15')(conv2_4))

    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=3)
    conv1_5 = cbam_resblock(conv1_5, filters=n_filters[0])

    output_1 = Conv2D(num_class, (1, 1), padding='same', activation='sigmoid', kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4), name='output_1')(conv1_2)
    output_2 = Conv2D(num_class, (1, 1), padding='same', activation='sigmoid', kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4), name='output_2')(conv1_3)
    output_3 = Conv2D(num_class, (1, 1), padding='same', activation='sigmoid', kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4), name='output_3')(conv1_4)
    output_4 = Conv2D(num_class, (1, 1), padding='same', activation='sigmoid', kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4), name='output_4')(conv1_5)

    if deep_supervision:
        outputs = [output_1,
                   output_2,
                   output_3,
                   output_4]
    else:
        outputs = [output_4]

    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == "__main__":
    shape = (288, 384, 3)
    model = build_model(shape, deep_supervision=True)
    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)
