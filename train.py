
import numpy as np
from utils import *
from keras.callbacks import ModelCheckpoint
import os
import time
import pickle

from tensorflow import set_random_seed
set_random_seed(3)


def train_model(params, cv, images, masks, pretrained_model, gpu, results_path, en_test):
    time_start = time.clock()

    # Fix random seed
    seed = 1
    np.random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if not os.path.exists(results_path + '/firstval' + str(cv['val'][0])):
        os.mkdir(results_path + '/firstval' + str(cv['val'][0]))

    # Load data
    list_train = cv['train']
    list_val = cv['val']
    if en_test:
        list_test = cv['test']

    train_images = images[list_train]
    train_masks = masks[list_train]
    val_images = images[list_val]
    val_masks = masks[list_val]
    if en_test:
        test_images = images[list_test]
        test_masks = masks[list_test]

    # Normalize data
    norm_params = {}
    norm_params['mu'] = np.mean(train_images)
    norm_params['sigma'] = np.std(train_images)
    pickle.dump(norm_params, open(results_path + '/firstval' + str(cv['val'][0]) + '/norm_params.p', "wb"))
    train_images = (train_images - norm_params['mu']) / norm_params['sigma']
    val_images = (val_images - norm_params['mu']) / norm_params['sigma']
    if en_test:
        test_images = (test_images - norm_params['mu']) / norm_params['sigma']

    # Add dimension for channel
    if not params['en_online']:
        train_images = np.expand_dims(train_images, axis=-1)
        train_masks = np.expand_dims(train_masks, axis=-1)
        val_images = np.expand_dims(val_images, axis=-1)
        val_masks = np.expand_dims(val_masks, axis=-1)
        if en_test:
            test_images = np.expand_dims(test_images, axis=-1)
            test_masks = np.expand_dims(test_masks, axis=-1)

    # Train model and save best
    if pretrained_model != 0:
        model = pretrained_model
    elif params['model'] == 'unet_2d':
        model = unet_2d(params)

    model_checkpoint = ModelCheckpoint(results_path + '/firstval' + str(cv['val'][0]) + '/weights.h5',
                                       verbose=1,
                                       monitor='val_' + params['loss'],
                                       save_best_only=False,
                                       save_weights_only=True,
                                       period=2)

    hist = model.fit(train_images,
                     train_masks,
                     batch_size=params['batch_size'],
                     nb_epoch=params['nb_epoch'],
                     verbose=2,
                     shuffle=True,
                     validation_data=(val_images, val_masks),
                     callbacks=[model_checkpoint])

    # Save training stats
    train_time = (time.clock() - time_start)
    np.save(results_path + '/firstval' + str(cv['val'][0]) + '/train_time.npy', train_time)
    save_history(hist.history, params, cv, results_path)



