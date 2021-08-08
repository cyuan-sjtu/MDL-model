from keras.models import *
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
from keras import initializers
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
#FC
def fcLSTM(pretrained_weights=None, input_size=(32, 64, 64, 3)):
    inputs = Input(input_size)
    paddings = tf.constant([[0, 0], [0,0],[1, 1], [1, 1], [0, 0], [0, 0]])

    [inputsct , inputspet,inputtemp] = Lambda(tf.split, arguments={'axis': 4, 'num_or_size_splits': 3})(inputs)

    conv1ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        inputsct)
    conv1ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv1ct)
    pool1ct = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1ct)
    conv2ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool1ct)
    conv2ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv2ct)
    pool2ct = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2ct)
    conv3ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool2ct)
    conv3ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv3ct)
    pool3ct = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3ct)
    conv4ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool3ct)
    conv4ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(conv4ct)
    pool4ct = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4ct)

    conv1pet = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        inputspet)
    conv1pet = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        conv1pet)
    pool1pet = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1pet)
    conv2pet = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        pool1pet)
    conv2pet = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        conv2pet)
    pool2pet = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2pet)
    conv3pet = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        pool2pet)
    conv3pet = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        conv3pet)
    pool3pet = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3pet)
    conv4pet = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        pool3pet)
    conv4pet = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        conv4pet)
    pool4pet = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4pet)

    comerge1_temp = concatenate([pool4ct, pool4pet], axis=4)
    poolctexp_temp = Lambda(expand_dim_backend, arguments={'dim': (5)})(pool4ct)
    poolpetexp_temp = Lambda(expand_dim_backend, arguments={'dim': (5)})(pool4pet)

    comerge2_temp = concatenate([poolctexp_temp, poolpetexp_temp], axis=5)
    input_mm = Lambda(tf.pad, arguments={'paddings': (paddings), 'mode': ("CONSTANT")})(comerge2_temp)
    input_mm = Lambda(tf.transpose, arguments={'perm': ([0, 1, 5, 2, 3, 4])})(input_mm)

    comerge2con_temp = TimeDistributed(Conv3D(filters=64, kernel_size=[2,3,3],
                         kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None),
                         kernel_regularizer=regularizers.l2(0.1),
                         bias_initializer='zeros', padding='valid', activation='relu'))(input_mm)
    colearn_out_temp = Lambda(tf.squeeze, arguments={'axis': (2)})(comerge2con_temp)
    conj1 = concatenate([comerge1_temp, colearn_out_temp], axis=4)
    reshapecom = Reshape((1,32*16*192))(conj1)
    reshapecom = Lambda(tf.squeeze, arguments={'axis': (1)})(reshapecom)
    Dense1 = Dense(3, activation= 'sigmoid', input_dim= 2, use_bias= True)(reshapecom)

    model = Model(input=inputs, output=Dense1)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
def expand_dim_backend(x,dim):
    xe = K.expand_dims(x, dim)
    return xe