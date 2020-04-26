# General libaries
import pandas as pd
import numpy as np
import os
import pickle

# Preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

# Neural network related
import keras
from keras import models
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Input, GlobalAveragePooling2D
from keras import layers
from keras.preprocessing.image import ImageDataGenerator # To create an image generator to create batches of images
from keras.preprocessing import image # To change images to an np array AND visualize the image
from keras import optimizers # to optimize
from keras.models import load_model # Load model
from keras.callbacks import ModelCheckpoint # To save best model
from tensorflow.keras import regularizers

from keras.utils.vis_utils import plot_model # To plot models
import pydot # To plot models

# Visualizing/visualizing activation of conv layers
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg

# To clear ram
from tensorflow.keras import backend as K
K.clear_session()

# To get information about ram
import multiprocessing

# Transfer Learning
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input



pd.set_option('max_colwidth', None)
pd.set_option("display.max_rows", 100)

np.seterr(divide='ignore', invalid='ignore')