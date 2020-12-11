# basic_unet
This repository shows an example of code for the segmentation of 2D images with 2D u-net in keras. The code supports two classes (i.e., foreground and background).

Get started by creating a folder named "sample". In this folder, insert your images and labels in the "numpy array" format and with the following size (nb samples, width, height). The labels should be binary masks (1 for the foreground and 0 for the background). You can launch the training and prediction from the run.py file.

The parameters for the network training can be updated in the run.py file. Training, validation and test images should be concatenated in the single image file (nb samples = nb training sample + nb validation samples + nb test samples). The model automatically performs a cross-validation and saves the corresponding model weights, learning curves, training time, normalisation parameters, in separate folders referenced by the index of the first validation sample. Those folders also contain the predictions obtained with the different models. The predictions are concatenated to reach the same size as the labels and saved in the same folder as the simulation parameters. 

An additional input channel can be added to the 2D U-net model in order to reproduce the results presented in the following conference publication: https://link.springer.com/chapter/10.1007/978-3-030-01449-0_32

This code can be used to partially reproduce the results of the following conference publication (only the 2D part): https://link.springer.com/chapter/10.1007/978-3-030-43195-2_12
