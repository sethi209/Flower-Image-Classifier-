# Flower-Image-Classifier-
This repository contains scripts for training and using a flower image classifier. It includes:

train.py: a script for training a pre-trained model on a dataset of flower images, and saving the trained model checkpoint.
predict.py: a script for predicting the class of a flower image using the trained model checkpoint.
utils.py: a script containing utility functions for loading and preprocessing flower images.
functions.py: a script containing functions necessary to execute the train.py and predict.py scripts.

Installation
Before using this repository, you need to install the following dependencies:

Python 3.x
PyTorch
torchvision
PIL

Usage
Training
To train the model, you need to run the train.py script. The script requires the following arguments:

data_dir: the path to the directory containing the training images.
checkpoint_path: the path to save the trained model checkpoint.
--arch: the pre-trained model architecture to use (default is 'vgg16').
--learning_rate: the learning rate for the optimizer (default is 0.001).
--hidden_units: the number of hidden units in the classifier layer (default is 512).
--epochs: the number of training epochs (default is 10).
--gpu: use GPU for training (default is False).

Prediction
To predict the class of a flower image, you need to run the predict.py script. The script requires the following arguments:

input: the path to the input image.
checkpoint: the path to the trained model checkpoint.
--top_k: the number of top classes to display (default is 5).
--category_names: the path to the JSON file containing category names (default is 'cat_to_name.json').
--gpu: use GPU for prediction (default is False).

Acknowledgments
This project was completed as part of the Udacity AI Programming with Python Nanodegree Program. The dataset used in this project is the 102 Category Flower Dataset.
