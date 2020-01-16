import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import scipy.io
import cv2

def GetDefaultParameters():
    '''
    Create a dictionary of parameters which will be used along the process
    :return: Dictionary of parameters
    '''
    path = r'C:\Users\BIGVU\Desktop\Yoav\University\computerVisionTask2\FlowerData'
    test_indices = list(range(301,473))
    labels = scipy.io.loadmat(r'C:\Users\BIGVU\Desktop\Yoav\University\computerVisionTask2\FlowerData\FlowerDataLabels.mat')["Labels"]
    image_size = (224,224)
    split = 0.2
    clusters = 40
    svm_c = 100
    degree = 3
    kernel = 'rbf'
    gamma = 5
    step_size = 6
    bins = clusters
    validate = False
    parameters = {"path":path,"Labels":labels,"test_indices":test_indices,"validate":validate,"image_size":image_size,"Split":split,"clusters":clusters,"step_size":step_size,"bins":bins, "svm_c":svm_c,"kernel":kernel,"gamma":gamma,'degree':degree}
    return parameters


def load_data(params):
    '''
    Loads the data
    :param path: data location on the PC
    :param params: desired size measure for all images
    :return: list of images raw data and list of its labels
    '''
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for i in range(1,params["Labels"].size+1):
        img = params['path'] + "\\"+str(i)+".jpeg"
        raw = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        sized = cv2.resize(raw, params["image_size"])  # resizing the data
        if i not in params['test_indices']:
            train_images.append(sized)
            train_labels.append(params['Labels'][0][i-1])
        else:
            test_images.append(sized)
            test_labels.append(params['Labels'][0][i-1])
    train_images = np.asarray(train_images)
    plt.imshow(train_images[54][:,:,[2,1,0]])
    plt.show()
    print("Data loading complete!")
    return train_images,train_labels,test_images,test_labels

def train_model(params,data,labels):
    '''
    Here we need to train only the final layer of the ResNet50v2. You do this by stating "include_top=False" we also want
    the weights to be trained so that would be "weights='imagenet'", we train using train_images and train_labels.

    :param params:
    :param data:
    :param labels:
    :return: Returns a trained Resnet50v2 model, with a new top layer
    '''


def test(model,data):
    '''
    Here we test, using model on the test data. Simply predict using the trained model.

    :param model:
    :param data:
    :param labels:
    :return: Returns a vector of predictions for the test data
    '''

def evaluate(predicts, probabilities, real,params):
    '''
    Here we evaluate the accuracy received by our model on the test set.

    :param predicts:
    :param probabilities:
    :param real:
    :param params:
    :return: accuracy/error score
    '''


def reportResults(error,real,params):
    '''
    Here we report the different things we were asked. Accuracy/Error, Confusion Matrix, Worst Images? If so, we need to
    Think if this function is the most convenient location to do so.

    :param error:
    :param real:
    :param params:
    :return:
    '''


def validation(params,param_to_validate,possible_values):
    '''
    Not sure yet how this function will look like. Once we understand how the different parameters look like, we can
    better give a pseudo code explanation of what needs to be here. Also, we might need to put the curve-recall curve
    here, which is a type of parameter, need to make sure this function knows how to deal with it.

    execution of the validation process
    :param params: dictionary of parameters
    :param param_to_validate: a hyper parameter which we want to examine
    :param possible_values: the hyper parameter range of values we wants to examine in an iterable object
    :return: Does not return anything
    '''

keras = tf.keras


params = GetDefaultParameters()
train_images,train_labels,test_images,test_labels = load_data(params)
