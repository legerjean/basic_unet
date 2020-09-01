from utils import *
from train import *
import numpy as np

gpu = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def predict_model(images, params, nb_classes, cv, results_path, en_save=True):
    
    # Get model
    if params['model'] == 'unet_2d':
        model = unet_2d(params)
    model.load_weights(results_path + '/firstval' + str(cv['val'][0]) + '/weights.h5')
    
    # Load normalization parameters
    pickle_in = open(results_path + '/firstval' + str(cv['val'][0]) + '/norm_params.p', "rb")
    norm_params = pickle.load(pickle_in)

    # Perform prediction
    sh = images.shape
    if nb_classes > 1:
        predictions = np.zeros((sh[0], sh[1], sh[2], nb_classes))
    else:
        predictions = np.zeros((sh[0], sh[1], sh[2]))
    predictions_thr = np.copy(predictions)

    for i in range(sh[0]):
        test_inputs_i = (images[i, :, :] - norm_params['mu']) / norm_params['sigma']
        test_inputs_i = np.expand_dims(test_inputs_i, axis=-1)
        test_inputs_i = np.expand_dims(test_inputs_i, axis=0)
        test_predictions_i = model.predict(test_inputs_i, batch_size=params['batch_size'], verbose=0)
        predictions[i] = np.squeeze(test_predictions_i)

    predictions_thr[predictions > 0.5] = 1
    predictions_thr = predictions_thr.astype(np.uint8)
    
    if en_save:
        np.save(results_path + '/firstval' + str(cv['val'][0]) + '/predictions.npy', predictions_thr)