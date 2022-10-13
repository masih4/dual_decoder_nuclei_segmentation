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
#####################################################################################
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
#####################################################################################
# Combination of Dice and binary cross entophy loss function that is used in this baseline segmentation
def bce_dice_loss(y_true, y_pred):
    return 0.5 * tensorflow.keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
#######################################################################################
# Mean Iou metrics
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2) # 2 is number of classes
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
#######################################################################################################
def binary_unet( IMG_CHANNELS, LearnRate):
    inputs = Input((None, None, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (p2)
    c3 = Dropout(0.1) (c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (p3)
    c4 = Dropout(0.1) (c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (p4)
    c5 = Dropout(0.1) (c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (u6)
    c6 = Dropout(0.1) (c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (u7)
    c7 = Dropout(0.1) (c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9) # for binary

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer = Adam(lr=LearnRate), loss= bce_dice_loss , metrics=[dice_coef]) #for binary

    #model.summary()
    return model


