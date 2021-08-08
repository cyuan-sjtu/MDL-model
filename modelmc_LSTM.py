from keras.models import *
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
from keras import initializers
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def mbLSTM(pretrained_weights=None, input_size=(32, 64, 64, 3)):
    inputs = Input(input_size)

    [inputsct , inputspet,inputtemp] = Lambda(tf.split, arguments={'axis': 4, 'num_or_size_splits': 3})(inputs)
    inputs_temp = concatenate([inputsct, inputspet], axis=4)
    conv1ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(inputs_temp)
    pool1ct = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1ct)
    conv2ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool1ct)
    pool2ct = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2ct)
    conv3ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool2ct)
    pool3ct = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3ct)
    conv4ct = TimeDistributed(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))(pool3ct)
    pool4ct = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4ct)

    pool4ct =Reshape((32,4*4*64))(pool4ct)

    LSTM1 = LSTM(64, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)(pool4ct)
    Dense1 = Dense(3, activation= 'sigmoid', input_dim= 2, use_bias= True)(LSTM1)

    model = Model(input=inputs, output=Dense1)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
