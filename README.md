# Deep Low Light Image Enhancement

This work considers the low light image enhancement problem of recovering an enhanced normal light version of a low contrast image which suffers from poor visibilty.
Image enhancement is an inherently ill-posed problem since a given low light image can have many possible normal light equivalents.
We seek to explore, compare and contrast various deep learning based approaches to the problem in supervised and unsupervised settings.
In particular we deal with the following 3 methods

1. Contrast Limited Adaptive Histogram Equalization (CLAHE): A conventional computer vision technique that adaptively equalized fized window regions in the image to improve the image contrast.
2. [EnlightenGAN](https://arxiv.org/abs/1906.06972): A GAN based low light image enhancement method that works without paired supervision.
3. Auto-encoders: U-Net based convolutional auto-encoder architectures with various ablations. 

# Programmer's Guide

* The codebase structure and base classes for the auto-encoder based experiments are derived from [this](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.
* This structure makes it easy to abstract away common contructs of dataset loading as well as model training, evaluation and visualization.
* In particular, we adopt the base classes for different entities (such as models and dataloaders) and the options handler for all our experiments with the U-Net auto-encoders.
* Based on this structure, we implement a model class for the auto-encoder (in *models/autoencoder\_model.py*) which includes code for initializing the model along with training and testing it based on custom options.
* *options/* contains various different options to select hyper-parameters and training and test setting along with paths to store the results and read the datasets as well as which models to use, number of epochs, etc.
* We also implement a generic dataloader for any dataset having training/validation/testing data (see *data/trainval\_dataloader.py*).
* The LoL dataset can be loaded by our custom dataloader defined in *data/lol\_dataset.py*.
* The scripts *train.py* and *test.py* can be used to train and test an instance of the model with specific commandline options.
* We experiment with multiple variations of the auto-encoder model while varying loss terms (RMSE, SSIM) as well as network architectures (bilinear upsampling, transpose convolution).
* We provide the scripts used to train and test these configurations in the *scripts/* directory.
* In particular, *ex3\_train.py* and *ex3\_test.py* can be used to reproduce the results from our report.
* In order to train a new model instance, simply create a bash script invoking *train.py* with the desired commandline parameters.

Trained models can be tested by invoking the script *test.py* with the desired commandline options specifying the epoch to use for testing along with the directory path to store the results.

# Acknowledgements

Website: https://pkmandke.github.io/ece5554-website/

Template used: https://github.com/yenchiah/project-website-template