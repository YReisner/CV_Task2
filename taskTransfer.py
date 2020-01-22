from operator import itemgetter

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
    path = r'D:\University\CV Task 2\FlowerData'
    test_indices = list(range(301,473))
    labels = scipy.io.loadmat(path + '\FlowerDataLabels.mat')["Labels"]
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
        final = sized / 255.0
        if i not in params['test_indices']:
            train_images.append(final)
            train_labels.append(params['Labels'][0][i-1])
        else:
            test_images.append(final)
            test_labels.append(params['Labels'][0][i-1])
    train_images = np.asarray(train_images)
    train_labels = np.asarray(train_labels)
    #plt.imshow(train_images[54][:,:,[2,1,0]])
    #plt.show()
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

    base_model = keras.applications.resnet_v2.ResNet50V2(include_top = False, weights = 'imagenet', input_shape = (224,224,3)) #load the base model
    base_model.trainable = False #Freezing the existing weights
    data = tf.cast(data,tf.float32) # Turn the train image data to a tf object
    labels = tf.cast(labels,tf.int8) # Same for labels
    train = data[0:240] # Arbitrarily choose train data, with bad practice of not shuffling (80%)
    validation = data[240:300] # rest is validation
    train_labels = labels[0:240]
    validation_labels = labels[240:300]
    '''
    train = tf.data.Dataset.from_tensor_slices((train,train_labels)) # turn the data + labels to tensors we can work with
    validation = tf.data.Dataset.from_tensor_slices((validation,validation_labels)) # same for validation
    train_batches = train.batch(30,drop_remainder=True) # We need to work with batches, otherwise memory won't take it (this is how everyone does it)
    validation_batches = validation.batch(30,drop_remainder=True) # same for validation
    for image_batch, label_batch in train_batches.take(1): # I just need the first image data batch from train to create the layers we need
        pass
    '''
    train_features = base_model(train[0:30]) # create final ResNet features to jump-start the sequential layers

    global_max_layer = keras.layers.GlobalMaxPooling2D() #Adding a convolutional layer to create somthing a global layer can handle
    feature_batch_average = global_max_layer(train_features) # convolutional layer that aggregates high dimensional data to a vector
    print(feature_batch_average.shape)
    prediction_layer = keras.layers.Dense(1, activation='sigmoid') # take vector and create predictions
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    model = keras.Sequential([
        base_model,
        global_max_layer,
        prediction_layer
    ]) # add sequntial layers to ResNet to create the sequential model


    base_learning_rate = 0.001 # hyper parameter
    model.compile(optimizer=keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy']) # define the learning parameters

    print(model.summary()) # how does our sequential model looks like?

    initial_epochs = 10 #number of time running over all the data in training
    validation_steps = 2 # this is for the evaluation. If we had more data, we could raise this for better results

    loss0, accuracy0 = model.evaluate(validation,validation_labels,batch_size=30,steps=validation_steps) # give the accuracy before training

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(train, train_labels,batch_size=30,
                        epochs=initial_epochs,
                        validation_data=(validation,validation_labels)) #train the model, and retain the results at every step

    # Just for printing the results, will be moved later to the proper function, I guess.
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print("the loss vector is")
    print(loss)
    print("the validation loss is")
    print(val_loss)

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    return model


def test(model,data):
    '''
    Here we test, using model on the test data. Simply predict using the trained model.

    :param model:
    :param data:
    :param labels:
    :return: Returns a vector of predictions for the test data
    '''
    probabilities = np.array(model.predict_proba(data)) # computing the probabilities
    predictions = np.where(probabilities > 0.5, 1, 0) #computing the predictions from the model
    return predictions,probabilities

def errors(predictions,probabilities,test_images,test_labels):
    '''
    finds the type1 and type 2 errors in the test data
    :param predictions:
    :param probabilities:
    :param params:
    :return: type_1: a vector of the 5 worse type 1 errors
            type_2: a vector of the 5 worse type 2 errors
    '''
    type_1 = []
    type_2 = []

    for i in range(len(predictions)):
        # miss detection - the alghorithem thought its not a flower  but it is
        if predictions[i] == 0 and test_labels[i] == 1:
            type_1.append(probabilities[i],i)

        # false alarm - the alghorithem thought its a flower but it is not
        if predictions[i] == 1 and test_labels[i] == 0:
            type_2.append(probabilities[i],i)

    ## subset the 5 worse errors from type 1,2
    type_1_sorted = sorted(type_1, reverse=True, key=itemgetter(0))
    type_2_sorted = sorted(type_2, reverse=True, key=itemgetter(0))
    five_type_1 = type_1_sorted[0:5]
    five_type_2 = type_2_sorted[0:5]
    print("type 1 errors")
    print(five_type_1)
    # need to save the images
    return five_type_1, five_type_2

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

np.random.seed(42)
params = GetDefaultParameters()
train_images,train_labels,test_images,test_labels = load_data(params)
model = train_model(params,train_images,train_labels)
pred_prob, pred, score = test(model,test_images)
