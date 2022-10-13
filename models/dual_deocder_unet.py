import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
#####################################################################################
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
#####################################################################################
def bce_dice_loss(y_true, y_pred):
    return 0.5 * tensorflow.keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
#####################################################################################################################
def mse_score(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
#####################################################################################################################
def dual_decoder_unet_binary(IMG_CHANNELS, LearnRate):
    inputs = Input((None, None, IMG_CHANNELS))
    #encoder
    s = Lambda(lambda x: x / 255)(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)


    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.1)(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    ## decoder for dis unet
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs1 = Conv2D(1, (1, 1), activation='linear', name='output_dis')(c9)


    ## decoder for segmentation unet
    u6_seg = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6_seg = concatenate([u6_seg, c4])
    c6_seg = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6_seg)
    c6_seg = Dropout(0.2)(c6_seg)
    c6_seg = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6_seg)

    u7_seg = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6_seg)
    u7_seg = concatenate([u7_seg, c3])
    c7_seg = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7_seg)
    c7_seg = Dropout(0.2)(c7_seg)
    c7_seg = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7_seg)

    u8_seg = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7_seg)
    u8_seg = concatenate([u8_seg, c2])
    c8_seg = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8_seg)
    c8_seg = Dropout(0.1)(c8_seg)
    c8_seg = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8_seg)

    u9_seg = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8_seg)
    u9_seg = concatenate([u9_seg, c1], axis=3)
    c9_seg = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9_seg)
    c9_seg = Dropout(0.1)(c9_seg)
    c9_seg = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9_seg)

    outputs2 = Conv2D(1, (1, 1), activation='sigmoid', name='output_seg')(c9_seg)

    model_dual_path = models.Model(inputs=[inputs], outputs=[outputs1, outputs2])
    model_dual_path.compile(optimizer=Adam(lr=LearnRate),
                            loss={'output_dis': 'mean_squared_error', 'output_seg': bce_dice_loss},
                            loss_weights=  {'output_dis': 1.0, 'output_seg': 1.0},
                            metrics={'output_seg':dice_coef, 'output_dis':mse_score})
    model_dual_path.summary()
    return model_dual_path


