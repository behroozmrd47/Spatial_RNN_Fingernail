from keras.models import Model
from keras.layers import *
from src.models.network.Spatial_RNN_2D import SpatialRNN2D


def unet2D_res_srnn_8_128(img_rows, img_cols, img_chns, pretrained_weight_path=False):
    inputs = Input((img_rows, img_cols, img_chns))  # (256,256,1)
    conv11 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv11')(
        inputs)  # (256,256,32)
    conc11 = concatenate([inputs, conv11], axis=3, name='conc11')  # (256,256,33)
    conv12 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv12')(
        conc11)  # (256,256,32)
    conc12 = concatenate([inputs, conv12], axis=3, name='conc12')  # (256,256,33)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conc12)  # (128,128,33)

    conv21 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv21')(
        pool1)  # (128,128,64)
    conc21 = concatenate([pool1, conv21], axis=3, name='conc21')  # (128,128,97)
    conv22 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv22')(
        conc21)  # (128,128,64)
    conc22 = concatenate([pool1, conv22], axis=3, name='conc22')  # (128,128,97)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conc22)  # (64,64,97)

    conv31 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv31')(
        pool2)  # (64,64,128)
    conc31 = concatenate([pool2, conv31], axis=3, name='conc31')  # (64,64,225)
    conv32 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv32')(
        conc31)  # (64,64,128)
    conc32 = concatenate([pool2, conv32], axis=3, name='conc32')  # (64,64,225)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conc32)  # (32,32,225)

    conv41 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv41')(
        pool3)  # (32,32,256)
    conc41 = concatenate([pool3, conv41], axis=3, name='conc41')  # (32,32,481)
    conv42 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv42')(
        conc41)  # (32,32,256)
    conc42 = concatenate([pool3, conv42], axis=3, name='conc42')  # (32,32,481)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conc42)  # (16,16,481)

    conv51 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv51')(
        pool4)  # (16,16,512)
    conc51 = concatenate([pool4, conv51], axis=3, name='conc51')  # (16,16,993)
    conv52 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv52')(
        conc51)  # (16,16,512)
    conc52 = concatenate([pool4, conv52], axis=3, name='conc52')  # (16,16,993)

    conv_tr_6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='conv_tr_6')(
        conc52)  # (32,32,256)
    conc60 = concatenate([conv_tr_6, conv42], axis=3, name='conc60')  # (32,32,512)

    conv61 = Conv2D(64, 3, activation='relu', padding='same', name='conv61')(conc60)  # (32,32,256)
    conc61 = concatenate([conc60, conv61], axis=3, name='conc61')  # (32,32,768)
    conv62 = Conv2D(64, 3, activation='relu', padding='same', name='conv62')(conc61)  # (32,32,256)
    conc62 = concatenate([conc60, conv62], axis=3, name='conc62')  # (32,32,768)

    conv_tr_7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc62)  # (64,64,128)
    rnn70 = SpatialRNN2D(rnn_seq_length=32, name='rnn70')(conv32)  # (64, 64, 512)
    conv70 = Conv2D(32, 1, activation='relu', padding='same', kernel_initializer='he_normal', name='conv70')(
        rnn70)  # (64,64,128)
    conc70 = concatenate([conv_tr_7, conv70], axis=3, name='conc70')  # (64,64,256)

    conv71 = Conv2D(32, 3, activation='relu', padding='same', name='conv71')(conc70)  # (64,64,128)
    conc71 = concatenate([conc70, conv71], axis=3, name='conc71')  # (64,64,384)
    conv72 = Conv2D(32, 3, activation='relu', padding='same', name='conv72')(conc71)  # (64,64,128)
    conc72 = concatenate([conc70, conv72], axis=3, name='conc72')  # (64,64,384)

    conv_tr_8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', name='conv_tr_8')(
        conc72)  # (128,128,64)
    rnn80 = SpatialRNN2D(rnn_seq_length=64, name='rnn80')(conv22)  # (128, 128, 256)
    conv80 = Conv2D(16, 1, activation='relu', padding='same', kernel_initializer='he_normal', name='conv80')(
        rnn80)  # (128,128,64)
    conc80 = concatenate([conv_tr_8, conv80], axis=3, name='conc80')  # (128,128,128)

    conv81 = Conv2D(16, 3, activation='relu', padding='same', name='conv81')(conc80)  # (128,128,64)
    conc81 = concatenate([conc80, conv81], axis=3, name='conc81')  # (128,128,192)
    conv82 = Conv2D(16, 3, activation='relu', padding='same', name='conv82')(conc81)  # (128,128,64)
    conc82 = concatenate([conc80, conv82], axis=3, name='conc82')  # (128,128,192)

    conv_tr_9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(conc82)  # (256,256,32)
    conc90 = concatenate([conv_tr_9, conv12], axis=3, name='conc90')  # (256,256,64)

    conv91 = Conv2D(8, 3, activation='relu', padding='same', name='conv91')(conc90)  # (256,256,32)
    conc91 = concatenate([conc90, conv91], axis=3, name='conc91')  # (256,256,96)
    conv92 = Conv2D(8, 3, activation='relu', padding='same', name='conv92')(conc91)  # (256,256,32)
    conc92 = concatenate([conc90, conv92], axis=3, name='conc92')  # (256,256,96)

    conv10 = Conv2D(1, 1, activation='sigmoid', name='conv10')(conc92)  # (256,256,1)

    model = Model(inputs, conv10, name='Unet_Res_SRNN_8x128')

    if pretrained_weight_path:
        model.load_weights(pretrained_weight_path)

    return model
