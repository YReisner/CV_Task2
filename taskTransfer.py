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

def train_model(params):
    '''
    Here we need to
    :param params:
    :return:
    '''





keras = tf.keras


params = GetDefaultParameters()
train_images,train_labels,test_images,test_labels = load_data(params)
