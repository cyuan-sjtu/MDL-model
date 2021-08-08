from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
from keras import initializers
from keras import regularizers

#PET
def cnn_pet_LSTM(pretrained_weights=None, input_size=(32, 64, 64, 3)):
    inputs = Input(input_size)
    paddings = tf.constant([[0, 0], [0, 0], [1, 1], [1, 1], [0, 0], [0, 0]])  # only pads dim 2 3 and 4 (h w and z)
    [inputsct, inputspet, inputtemp] = Lambda(tf.split, arguments={'axis': 4, 'num_or_size_splits': 3})(inputs)

    conv1pet = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        inputspet)
    pool1pet = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1pet)
    conv2pet = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        pool1pet)
    pool2pet = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2pet)
    conv3pet = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        pool2pet)
    pool3pet = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3pet)
    conv4pet = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        pool3pet)
    pool4pet = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4pet)

    reshape = Reshape((32, 4 * 4 * 64))(pool4pet)

    LSTM1 = LSTM(128, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
                 unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                 dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False,
                 go_backwards=False, stateful=False, unroll=False)(reshape)
    Dense1 = Dense(3, activation='sigmoid', input_dim=2, use_bias=True)(LSTM1)

    model = Model(input=inputs, output=Dense1)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
#CT
def cnn_ct_LSTM(pretrained_weights=None, input_size=(32, 64, 64, 3)):
    inputs = Input(input_size)
    paddings = tf.constant([[0, 0], [0, 0], [1, 1], [1, 1], [0, 0], [0, 0]])  # only pads dim 2 3 and 4 (h w and z)
    [inputsct, inputspet, inputtemp] = Lambda(tf.split, arguments={'axis': 4, 'num_or_size_splits': 3})(inputs)

    conv1ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(
        inputsct)
    pool1ct = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1ct)
    conv2ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool1ct)
    pool2ct = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2ct)
    conv3ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool2ct)
    pool3ct = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3ct)
    conv4ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool3ct)
    pool4ct = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4ct)

    reshape = Reshape((32, 4 * 4 * 64))(pool4ct)

    LSTM1 = LSTM(128, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
                 unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                 dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False,
                 go_backwards=False, stateful=False, unroll=False)(reshape)
    Dense1 = Dense(3, activation='sigmoid', input_dim=2, use_bias=True)(LSTM1)

    model = Model(input=inputs, output=Dense1)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
def expand_dim_backend(x, dim):
    xe = K.expand_dims(x, dim)
    return xe
