from keras.models import Model
from keras.layers import *


def unet2d_res_32_512(img_rows, img_cols, img_chns, pretrained_weight_path=False):
    inputs = Input((img_rows, img_cols, img_chns))  # (256,256,1)
    conv11 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)  # (256,256,32)
    conc11 = concatenate([inputs, conv11], axis=3)  # (256,256,33)
    conv12 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conc11)  # (256,256,32)
    conc12 = concatenate([inputs, conv12], axis=3)  # (256,256,33)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conc12)  # (128,128,33)

    conv21 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)  # (128,128,64)
    conc21 = concatenate([pool1, conv21], axis=3)  # (128,128,97)
    conv22 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conc21)  # (128,128,64)
    conc22 = concatenate([pool1, conv22], axis=3)  # (128,128,97)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conc22)  # (64,64,97)

    conv31 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)  # (64,64,128)
    conc31 = concatenate([pool2, conv31], axis=3)  # (64,64,225)
    conv32 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conc31)  # (64,64,128)
    conc32 = concatenate([pool2, conv32], axis=3)  # (64,64,225)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conc32)  # (32,32,225)

    conv41 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)  # (32,32,256)
    conc41 = concatenate([pool3, conv41], axis=3)  # (32,32,481)
    conv42 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conc41)  # (32,32,256)
    conc42 = concatenate([pool3, conv42], axis=3)  # (32,32,481)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conc42)  # (16,16,481)

    conv51 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)  # (16,16,512)
    conc51 = concatenate([pool4, conv51], axis=3)  # (16,16,993)
    conv52 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conc51)  # (16,16,512)
    conc52 = concatenate([pool4, conv52], axis=3)  # (16,16,993)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc52), conc42], axis=3)  # (32,32,737)  # (32,32,256)
    conv61 = Conv2D(256, 3, activation='relu', padding='same')(up6)  # (32,32,256)
    conc61 = concatenate([up6, conv61], axis=3)  # (32,32,993)
    conv62 = Conv2D(256, 3, activation='relu', padding='same')(conc61)  # (32,32,256)
    conc62 = concatenate([up6, conv62], axis=3)  # (32,32,993)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc62), conv32], axis=3)  # (64,64,256)
    conv71 = Conv2D(128, 3, activation='relu', padding='same')(up7)
    conc71 = concatenate([up7, conv71], axis=3)
    conv72 = Conv2D(128, 3, activation='relu', padding='same')(conc71)
    conc72 = concatenate([up7, conv72], axis=3)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc72), conv22], axis=3)
    conv81 = Conv2D(64, 3, activation='relu', padding='same')(up8)
    conc81 = concatenate([up8, conv81], axis=3)
    conv82 = Conv2D(64, 3, activation='relu', padding='same')(conc81)
    conc82 = concatenate([up8, conv82], axis=3)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc82), conv12], axis=3)
    conv91 = Conv2D(32, 3, activation='relu', padding='same')(up9)
    conc91 = concatenate([up9, conv91], axis=3)
    conv92 = Conv2D(32, 3, activation='relu', padding='same')(conc91)
    conc92 = concatenate([up9, conv92], axis=3)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conc92)

    model = Model(inputs, conv10, name='Unet_Res_32x512')

    if pretrained_weight_path:
        model.load_weights(pretrained_weight_path)

    return model
