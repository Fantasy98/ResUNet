from tensorflow.keras.regularizers import l2

from models.utils import *

dropout_rate = 0.5
act = "relu"


def standard_unit(inputs, stage, filters, kernel_size=(3, 3)):
    x = Conv2D(filters, kernel_size, activation=act, name='conv' + stage + '_1',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = Dropout(dropout_rate, name='dp' + stage + '_1')(x)
    x = Conv2D(filters, kernel_size, activation=act, name='conv' + stage + '_2',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate, name='dp' + stage + '_2')(x)

    return x


def unet_plus_plus(shape=(256, 256, 3), num_classes=1, deep_supervision=False):
    """
    UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation.
    Paper: https://arxiv.org/pdf/1807.10165.pdf & https://arxiv.org/pdf/1912.05074.pdf
    Code from: https://github.com/MrGiovanni/UNetPlusPlus
    """
    n_filters = [32, 64, 128, 256, 512]

    inputs = Input(shape=shape, name='main_input')

    conv1_1 = standard_unit(inputs, stage='11', filters=n_filters[0])
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', filters=n_filters[1])
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(n_filters[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=3)
    conv1_2 = standard_unit(conv1_2, stage='12', filters=n_filters[0])

    conv3_1 = standard_unit(pool2, stage='31', filters=n_filters[2])
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(n_filters[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=3)
    conv2_2 = standard_unit(conv2_2, stage='22', filters=n_filters[1])

    up1_3 = Conv2DTranspose(n_filters[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=3)
    conv1_3 = standard_unit(conv1_3, stage='13', filters=n_filters[0])

    conv4_1 = standard_unit(pool3, stage='41', filters=n_filters[3])
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(n_filters[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=3)
    conv3_2 = standard_unit(conv3_2, stage='32', filters=n_filters[2])

    up2_3 = Conv2DTranspose(n_filters[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=3)
    conv2_3 = standard_unit(conv2_3, stage='23', filters=n_filters[1])

    up1_4 = Conv2DTranspose(n_filters[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=3)
    conv1_4 = standard_unit(conv1_4, stage='14', filters=n_filters[0])

    conv5_1 = standard_unit(pool4, stage='51', filters=n_filters[4])

    up4_2 = Conv2DTranspose(n_filters[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=3)
    conv4_2 = standard_unit(conv4_2, stage='42', filters=n_filters[3])

    up3_3 = Conv2DTranspose(n_filters[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=3)
    conv3_3 = standard_unit(conv3_3, stage='33', filters=n_filters[2])

    up2_4 = Conv2DTranspose(n_filters[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=3)
    conv2_4 = standard_unit(conv2_4, stage='24', filters=n_filters[1])

    up1_5 = Conv2DTranspose(n_filters[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=3)
    conv1_5 = standard_unit(conv1_5, stage='15', filters=n_filters[0])

    nestnet_output_1 = Conv2D(num_classes, (1, 1), activation='sigmoid', name='output_1', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_classes, (1, 1), activation='sigmoid', name='output_2', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_classes, (1, 1), activation='sigmoid', name='output_3', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_classes, (1, 1), activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        outputs = [nestnet_output_1,
                   nestnet_output_2,
                   nestnet_output_3,
                   nestnet_output_4]
    else:
        outputs = [nestnet_output_4]

    model = Model(inputs=inputs, outputs=outputs, name="unet_plus_plus")

    return model


if __name__ == "__main__":
    model = unet_plus_plus()
    model.summary()
