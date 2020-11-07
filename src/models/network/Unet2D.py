from keras.models import Model
from keras.layers import *


def unet2d_8_128(img_rows, img_cols, img_chns, pretrained_weight_path=False):
    s = Input((img_rows, img_cols, img_chns))  # (256,256,1)
    c1 = Conv2D(8, 3, activation='relu', padding='same')(s)
    c1 = Conv2D(8, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)
    c2 = Conv2D(16, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(16, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)
    c3 = Conv2D(32, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(32, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)
    c4 = Conv2D(64, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(64, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D()(c4)
    c5 = Conv2D(128, 3, activation='relu', padding='same')(p4)
    c5 = Conv2D(128, 3, activation='relu', padding='same')(c5)
    u6 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c5)
    u6 = Concatenate(axis=3)([u6, c4])
    c6 = Conv2D(64, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(64, 3, activation='relu', padding='same')(c6)
    u7 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(c6)
    u7 = Concatenate(axis=3)([u7, c3])
    c7 = Conv2D(32, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(32, 3, activation='relu', padding='same')(c7)
    u8 = Conv2DTranspose(16, 2, strides=(2, 2), padding='same')(c7)
    u8 = Concatenate(axis=3)([u8, c2])
    c8 = Conv2D(16, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(16, 3, activation='relu', padding='same')(c8)
    u9 = Conv2DTranspose(8, 2, strides=(2, 2), padding='same')(c8)
    u9 = Concatenate(axis=3)([u9, c1])
    c9 = Conv2D(8, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(8, 3, activation='relu', padding='same')(c9)
    outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    model = Model(inputs=s, outputs=outputs, name='Unet_8x128')

    if pretrained_weight_path:
        model.load_weights(pretrained_weight_path)

    return model
