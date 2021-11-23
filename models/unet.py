from models.utils import *


def unet(shape=(256, 256, 3)):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation.
    Paper: https://arxiv.org/pdf/1505.04597.pdf
    Code from: https://github.com/zhixuhao/unet
    """
    # n_filters = [64, 128, 256, 512, 1024]
    n_filters = [32, 64, 128, 256, 512]

    inputs = Input(shape)

    conv1 = Conv2D(n_filters[0], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(n_filters[0], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = Conv2D(n_filters[1], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(n_filters[1], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    conv3 = Conv2D(n_filters[2], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(n_filters[2], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

    conv4 = Conv2D(n_filters[3], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(n_filters[3], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop4)

    conv5 = Conv2D(n_filters[4], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(n_filters[4], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(n_filters[3], (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))

    merge6 = concatenate([drop4, up6], axis=3)

    conv6 = Conv2D(n_filters[3], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(n_filters[3], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(n_filters[2], (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))

    merge7 = concatenate([conv3, up7], axis=3)

    conv7 = Conv2D(n_filters[2], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(n_filters[2], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(n_filters[1], (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))

    merge8 = concatenate([conv2, up8], axis=3)

    conv8 = Conv2D(n_filters[1], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(n_filters[1], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(n_filters[0], (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))

    merge9 = concatenate([conv1, up9], axis=3)

    conv9 = Conv2D(n_filters[0], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(n_filters[0], (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv9 = Conv2D(2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10, name="unet")

    return model


if __name__ == "__main__":
    model = unet()
    model.summary()
