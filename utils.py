from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, Conv2D, MaxPooling2D, \
    Conv2DTranspose, Dropout, Add, Lambda, multiply, SpatialDropout3D, SpatialDropout2D, LeakyReLU, BatchNormalization
import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.regularizers import l2
from keras.optimizers import Adam
import pickle
import os
import matplotlib.pyplot as plt
import csv


###############################
# Losses
###############################

def dice_loss_2d(y_true, y_pred):
    smooth = 1
    sh = tf.shape(y_true)
    y_true_f = tf.transpose(tf.reshape(y_true, [sh[0], sh[1] * sh[2]]))
    y_pred_f = tf.transpose(tf.reshape(y_pred, [sh[0], sh[1] * sh[2]]))
    intersection = tf.multiply(y_true_f, y_pred_f)
    intersection = tf.reduce_sum(intersection, 0)
    card_y_true = tf.reduce_sum(y_true_f, 0)
    card_y_pred = tf.reduce_sum(y_pred_f, 0)
    dices = tf.div(2 * intersection, card_y_true + card_y_pred + smooth)
    return -tf.reduce_mean(dices)


###############################
# Metrics
###############################

def dice_2d(y_true, y_pred):
    sh = tf.shape(y_true)
    y_true_f = tf.transpose(tf.reshape(y_true, [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_f = tf.transpose(tf.reshape(y_pred, [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_f = K.cast(K.greater(y_pred_f, 0.5), K.floatx())
    intersection = tf.multiply(y_true_f, y_pred_f)
    intersection = tf.reduce_sum(intersection, 0)
    card_y_true = tf.reduce_sum(y_true_f, 0)
    card_y_pred = tf.reduce_sum(y_pred_f, 0)
    dices = tf.div(2 * intersection, card_y_true + card_y_pred)
    return tf.reduce_mean(dices)


###############################
# Models
###############################

def unet_2d(params):
    nb_layers = params['n_layers']
    nb_features = params['n_feat_maps']

    # Input layer
    inputs = Input(batch_shape=(None, None, None, 1))

    # Encoding part
    skips = []
    x = inputs
    for i in range(nb_layers):
        # First conv, bn, ReLu, dropout block
        x = Conv2D(nb_features, (3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout2D(params['dropout'])(x) if params['dropout'] != 0 else x

        # Second conv, bn, ReLu, dropout block
        x = Conv2D(nb_features, (3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout2D(params['dropout'])(x) if params['dropout'] != 0 else x

        # Skip connection and maxpooling
        skips.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        nb_features = nb_features*2

    # Bottleneck
    x = Conv2D(nb_features, (3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
    x = BatchNormalization()(x) if params['bn'] else x
    x = LeakyReLU(alpha=0.0)(x)
    x = SpatialDropout2D(params['dropout'])(x) if params['dropout'] != 0 else x
    x = Conv2D(nb_features, (3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
    x = BatchNormalization()(x) if params['bn'] else x
    x = LeakyReLU(alpha=0.0)(x)
    x = SpatialDropout2D(params['dropout'])(x) if params['dropout'] != 0 else x

    # Decoding part
    for i in reversed(range(nb_layers)):
        nb_features = int(nb_features / 2)

        # Upsampling and concatenate
        x = concatenate([Conv2DTranspose(nb_features, (2, 2), strides=(2, 2), padding='same',
                                         kernel_initializer=params['init'], bias_initializer=params['init'],
                                         kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x),
                         skips[i]], axis=3)

        # First conv, bn, ReLu, dropout block
        x = Conv2D(nb_features, (3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout2D(params['dropout'])(x) if params['dropout'] != 0 else x

        # Second conv, bn, ReLu, dropout block
        x = Conv2D(nb_features, (3, 3), activation='linear', padding='same',
                   kernel_initializer=params['init'], bias_initializer=params['init'],
                   kernel_regularizer=l2(params['wd']), bias_regularizer=l2(params['wd']))(x)
        x = BatchNormalization()(x) if params['bn'] else x
        x = LeakyReLU(alpha=0.0)(x)
        x = SpatialDropout2D(params['dropout'])(x) if params['dropout'] != 0 else x

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid',
                     kernel_initializer=params['init'], bias_initializer=params['init'])(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    if params['loss'] == 'dice_loss_2d':
        model.compile(optimizer=Adam(params['lr']), loss=dice_loss_2d, metrics=[dice_2d])

    return model


###############################
# Misc
###############################

def save_history(hist, params, cv, results_path):
    x = range(1, len(hist['loss']) + 1)
    plt.figure(figsize=(12, 12))
    plt.plot(x, hist['dice_2d'], 'o-', label='train')
    plt.plot(x, hist['val_dice_2d'], 'o-', label='val')
    plt.legend(loc='upper left')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.savefig(results_path + '/firstval' + str(cv['val'][0]) + '/learning_curves.png')
    plt.close()


def params2name(params):
    results_name = ''
    for key in params.keys():
        results_name = results_name + key + '_' + str(params[key]) + '_'
    results_name = results_name[:-1]
    return results_name


def save_params(params, path):
    if not os.path.exists(path):
        os.mkdir(path)
    pickle.dump(params, open(path + '/params.p', "wb"))
    with open(path + '/params.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
            writer.writerow([key, value])


def cv_index_generator(params, nb_train_max, results_path, fold_index=0, en_print=True):
    fa = params['factor_augment']
    ind_trainvaltest = nb_train_max * fa + int((params['nVal'] + params['nTest']) * fa)
    ind_trainval = nb_train_max * fa + int(params['nVal'] * fa)
    ind_train = nb_train_max * fa
    listOfIndices = np.roll(np.arange(ind_trainvaltest), -fold_index * params['nVal'] * fa)
    trainList = listOfIndices[0:params['nTrain']]
    valList = listOfIndices[ind_train:ind_trainval:fa]
    testList = listOfIndices[ind_trainval:ind_trainvaltest:fa]
    cv = {'train': trainList, 'val': valList, 'test': testList, 'cvNum': fold_index}

    if en_print:
        print(cv['train'])
        print(cv['val'])
        print(cv['test'])

    if not os.path.exists(results_path + '/firstval' + str(cv['val'][0])):
        os.makedirs(results_path + '/firstval' + str(cv['val'][0]))
    pickle.dump(cv, open(results_path + '/firstval' + str(cv['val'][0]) + '/cv.p', "wb"))

    return cv