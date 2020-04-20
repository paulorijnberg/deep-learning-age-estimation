# General libaries
import pandas as pd
import numpy as np
import os
import pickle

# Preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix

# Neural network related
import keras
from keras import models
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Input
from keras import layers
from keras.preprocessing.image import ImageDataGenerator # To create an image generator to create batches of images
from keras.preprocessing import image # To change images to an np array AND visualize the image
from keras import optimizers # to optimize
from keras.models import load_model # Load model
from keras.callbacks import ModelCheckpoint # To save best model

from keras.utils.vis_utils import plot_model # To plot models
import pydot # To plot models
import seaborn as sn

# Visualizing/visualizing activation of conv layers
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import keract

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



def load_dataset(filename):
    """
    This function loads the dataset.
    """
    with open(filename, 'rb') as input_file:
        df = pickle.load(input_file)
        
    return df


def save_history_model(history, filename, storage_location):
    """
    This function pickles the history of a model and stores it
    at the storage_location with the given filename.
    """
    
    # Pickle the result
    location = os.path.join(storage_location, filename + '.pkl')
    
    with open(location, 'wb') as output_file:
        pickle.dump(history, output_file)
        
def plot_history(history, x_as, y_as, title, line1='accuracy', line2='val_accuracy'):
    """
    This function plots two lines, which as default are accuracy
    and val_accuracy.
    """
    
    history = history.history
    
    plt.plot(range(1, len(history[line1]) + 1), history[line1])
    plt.plot(range(1, len(history[line2]) + 1), history[line2])
    plt.xlabel(x_as)
    plt.ylabel(y_as)
    plt.title(title)
    plt.legend([line1, line2])
    plt.show()
    
def load_neural_network(model_path, weights_path):
    """
    Load the model.
    """
    
    # Load model
    model = load_model(model_path)
    
    # Load weights
    model.load_weights(weights_path)
    
    # Return model
    return model

def plot_image_activation(model_path, weights_path, img_path):
    """
    This images shows the activation of the last convolutional layer.
    """
    
    # Define model
    model = load_neural_network(model_path, weights_path)
    
    # Extract model_input_shape and relevant information
    model_input_shape = model.layers[0].input_shape
    
    # Define color mode
    if model_input_shape[-1] == 1:
        color_mode = 'grayscale'
        
    elif model_input_shape[-1] == 3:
        color_mode = 'rgb'
    
    # Extract target size
    target_size = model.layers[0].input_shape[1:3]
    
    # Load the image that we want to plot
    img = image.load_img(path        = img_path,
                         color_mode  = color_mode,
                         target_size = target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    if model_input_shape[-1] == 3:
        x = preprocess_input(x)
        
    
    # Make a prediction
    preds = model.predict(x)
    
    # Print predictions
    print('Predictions: {}.'.format(preds))
    
    
    # Get information of the prediction
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]
    
    for layer in model.layers:
        if 'conv' in layer.name: 
            last_conv_layer_name = layer.name
    last_conv_layer = model.get_layer(last_conv_layer_name)

    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    
    for i in range(last_conv_layer.output_shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    
    # Plot the result
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = .8
    superimposed_img = heatmap * hif + img

    output = os.path.join(os.getcwd(), 'test.jpeg')
    cv2.imwrite(output, superimposed_img)

    img = mpimg.imread(output)
    
    plt.imshow(img)
    plt.axis('off')    
    
    # Delete img after showing
    os.remove(output)
    
    
def plot_confusion_matrix(model_path, weights_path, generator):
    """
    Create a confusion matrix and outputs the mean class accuracy.
    """
    # Make prediction with model
    model  = load_neural_network(model_path,
                                 weights_path)
    
    y_pred = model.predict_generator(generator)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = generator.classes
    labels = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70']
    labels_numbered = [index for index, label in enumerate(labels)]
    
    ### Print mean class accuracy ####
    count_classes    = [[label, len([number for number in y_true if number == index])] for index, label in enumerate(labels_numbered)]
    
    # Count amount of y_true of each class
    all_label_counts = []
        
    for index, label in enumerate(count_classes):
        if index == 0: 
            all_label_counts.append([label[0], label[1]])
            
        else: 
            all_label_counts.append([label[0], label[1] + all_label_counts[index - 1][1]])
        
    # Create nested lists of each class and the predicted values
    y_pred_binned = []
    for index, label in enumerate(all_label_counts):
        if index == 0:
            y_pred_binned.append([label[0], y_pred[:label[1]]])

        else:
            y_pred_binned.append([label[0], y_pred[all_label_counts[index - 1][1]:label[1]]])
    
    
    # Calculate mean class accuracies
    y_pred_binned_mca = []
    for index, collection in enumerate(y_pred_binned):   
        len_of_index = len(collection[1][collection[1] == index])
        accuracy     = len_of_index/count_classes[index][1]

        y_pred_binned_mca.append([index, accuracy])

    # Print accuracies
    print('Mean class accuracies:\n')
    for index, class_accuracy in enumerate(y_pred_binned_mca):
        print('Class {} has an accuracy of {}.'
              .format(labels[index],
                      round(class_accuracy[1], 2)))
    
    ### Confusion matrix ###
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, 
                         index = [i for i in labels],
                         columns = [i for i in labels])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True,cmap="OrRd")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")
    
    
def plot_layers(model_path, best_val_score, img_path, target_size_img, color_mode_img):
    """
    Credits: https://github.com/philipperemy/keract
    """
    
    model = load_neural_network(model_path, best_val_score)
    
    counter = 0
    for layer in model.layers:
        if 'conv' in layer.name: 
            last_conv_layer_name = layer.name
            counter += 1
        
        if counter > 5:
            break
    
    
    img = image.load_img(img_path, target_size=(target_size_img), color_mode=color_mode_img)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    
    
    activations = keract.get_activations(model, img_tensor, layer_name=last_conv_layer_name)
    keract.display_heatmaps(activations, img_tensor)