from operator import itemgetter  # only needed for image error printing
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
import tensorflow as tf # Tensorflow 2.1 was used, sorry if that causes any issues.
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import cv2

'''
Dear user of this code,

We have written this code so that the only changes that need to be made are inside the function "GetDefaultParameters" 
just below this message. 
Please change "data_path" to your directory of the file "FlowerData"
Also, please change "test_indices" to the relevant indices you want to be treated as test. 
Otherwise, nothing is supposed to be changed. 
These changes can also be made on main, by simply redefining the dictionary values instead of simply re-assigning them. 
Please take into consideration that if "validate" is true, a cross validation process will begin on the parameter 
and with the values mentioned at the call to the validation function.

'''


def GetDefaultParameters():
    '''
    Create a dictionary of parameters which will be used along the process
    :return: Dictionary of parameters
    '''
    data_path = r'D:\University\CV Task 2\FlowerData' # change to your directory
    test_indices = list(range(301,473)) # list of indices for test
    labels = scipy.io.loadmat(data_path + '\FlowerDataLabels.mat')["Labels"]
    image_size = (224,224)
    split = 0.2
    input_shape = (224,224,3)
    activation ='sigmoid'
    test_size = 0.2
    validate = False
    momentum = 0.8
    learning_rate =0.05
    weights = 'imagenet'
    metrics =['accuracy']
    batch_size = 40
    loss = 'binary_crossentropy'
    epochs = 13
    test_batch_size = 32
    seed = 42
    decay = 0.5
    parameters = {"path":data_path,"Labels":labels,"test_indices":test_indices,"validate":validate,"image_size":image_size,
                  "Split":split,"input_shape":input_shape, "activation":activation,"test_size":test_size,"momentum":momentum,
                  "learning_rate":learning_rate,"decay":decay,"weights":weights,"loss":loss,"metrics":metrics,"batch_size":batch_size,"epochs":epochs,"test_batch_size":test_batch_size,"seed":seed}

    return parameters


def load_data(params):
    '''
    Loads the data, pre-process it and splits into train and test according to the test indices given
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
    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)
    print("Data loading complete!")
    return train_images,train_labels,test_images,test_labels

def base_train_model(params,data,labels):
    '''
   This is them most basic sequntial model we built over ResNet50V2, without a lot of parameters or tuning.
   This function allows us to access our starting point at any given time if needed.

    :param params: Default Parameters
    :param data: Train images
    :param labels: Train Labels
    :return: Returns a trained sequntial Resnet50v2 model, with a new top layer
    '''

    base_model = keras.applications.resnet_v2.ResNet50V2(include_top = False, weights = params["weights"], input_shape =params["input_shape"] ) #load the base model
    base_model.trainable = False #Freezing the existing weights

    array_data = np.array(data)
    array_labels = np.array(labels)

    train_x, val_x, train_y, val_y = train_test_split(array_data,array_labels,test_size=params["test_size"],random_state=params["seed"])


    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  #Adding a pooling layer to create something a global layer can handle
    prediction_layer = keras.layers.Dense(1, params["activation"]) # take vector and create predictions

    model = keras.Sequential([base_model,global_average_layer,prediction_layer]) # add sequntial layers to ResNet to create the sequential model

   # hyper parameter
    model.compile(optimizer=keras.optimizers.SGD(lr=0.001),
                  loss ='binary_crossentropy',metrics=['accuracy']) # define the learning parameters

    print(model.summary()) # how does our sequential model looks like?
    # train the model, and retain the results at every step
    history = model.fit(train_x, train_y,batch_size=32, epochs=13, shuffle=True, validation_data=(val_x,val_y),verbose=2)

    print("Model trained!")
    return model, history.history


def train_best_model(params,data,labels):
    '''
   This is the training function for the best sequential model after hyper parameter turning

    :param params: Default Parameters
    :param data: Train Images
    :param labels: Train Labels
    :return: Returns a trained squential model with Resnet50v2 as basis, with new layers we train
    '''

    base_model = keras.applications.resnet_v2.ResNet50V2(include_top = False, weights = params["weights"], input_shape =params["input_shape"] ) #load the base model
    base_model.trainable = False #Freezing the existing weights
    for layer in base_model.layers:
        layer.trainable = False
    array_data = np.array(data)
    array_labels = np.array(labels)

    train_x, val_x, train_y, val_y = train_test_split(array_data,array_labels,test_size=params["test_size"],random_state=params["seed"])

    train_features = base_model(train_x[0:1]) # create final ResNet features to jump-start the sequential layers
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  #Adding a pooling layer to create somthing a global layer can handle
    feature_batch_max = global_average_layer(train_features) # convolutional layer that aggregates high dimensional data to a vector
    dropout_layer = keras.layers.Dropout(0.5, seed=params["seed"])

    prediction_layer = keras.layers.Dense(1, params["activation"], kernel_regularizer=keras.regularizers.l2(params['decay'])) # take vector and create predictions

    model = keras.Sequential([base_model,global_average_layer,dropout_layer,prediction_layer]) # add sequntial layers to ResNet to create the sequential model

   # hyper parameter
    model.compile(optimizer=keras.optimizers.SGD(lr=params["learning_rate"],momentum = params["momentum"]),
                  loss =params["loss"],metrics=params["metrics"]) # define the learning parameters

    # train the model, and retain the results at every step
    history = model.fit(train_x, train_y,batch_size=params["batch_size"], epochs=params["epochs"], shuffle=True, validation_data=(val_x,val_y),verbose=2)

    print("Model Trained!")
    return model, history.history


def test(model,data,params):
    '''
    Here we test, using model on the test data. Simply predict using the trained model.

    :param model: The Sequential model we trained
    :param data: Test Images
    :return: Returns a vector of predictions (labels) and predictions (sigmoid output)  for the test data
    '''

    probabilities = np.array(model.predict_proba(data,batch_size=params["test_batch_size"],verbose=0)) # computing the probabilities
    predictions = np.where(probabilities > 0.5, 1, 0) #computing the predictions from the model
    return predictions,probabilities


def precision_recall_graph(predictions, test_y):
    '''
    This function  print precision vs recall graph
    :param predictions: Predictions of our model on the test images
    :param test_labels: Ground truth for test indices
    :return: Graphs the precision recall graph
    '''

    preds = np.array(predictions)
    test_labels = np.array(test_y)

    precision, recall, _ = precision_recall_curve(test_labels.ravel(), preds.ravel())

    plt.step(recall, precision, color='black', alpha=0.7, where='post')
    plt.fill_between(recall, precision, alpha=0.3, color='orange',)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision VS Recall Curve')
    plt.show()

    return


def errors(test_data,predictions,probabilities,test_labels):
    '''
     finds and prints the top 5 type1 and type2 errors in the test data
    :param predictions: test predictions
    :param probabilities: test probabilities
    :param test_labels:the real labels
    '''

    type_1 = []
    type_1_indices = []
    type_2 = []
    type_2_indices = []
    for i in range(len(predictions)):
        # miss detection - the algorithem thought its not a flower  but it is
        if predictions[i] == 0 and test_labels[i] == 1:
            type_1.append(probabilities[i])
            type_1_indices.append(i)
        # false alarm - the algorithem thought its a flower but it is not
        elif predictions[i] == 1 and test_labels[i] == 0:
            type_2.append(probabilities[i])
            type_2_indices.append(i)
    ## subset the 5 worse errors from type 1,2

    type_1_tuple = list(zip(type_1,type_1_indices))
    type_2_tuple = list(zip(type_2,type_2_indices))
    type_1_sorted = sorted(type_1_tuple, reverse=False, key=itemgetter(0))
    type_2_sorted = sorted(type_2_tuple, reverse=True, key=itemgetter(0))
    five_type_1 = type_1_sorted[0:5]
    five_type_2 = type_2_sorted[0:5]

    i = 1
    for tuple in five_type_1:
        plt.imshow(test_data[tuple[1]][..., ::-1])
        plt.title("The Number %d worst False Negative picture" % (i))
        plt.show()
        i += 1

    i = 1
    for tuple in five_type_2:
        plt.imshow(test_data[tuple[1]][..., ::-1])
        plt.title("The Number %d worst False Positive picture" % (i))
        plt.show()
        i += 1

    return


def evaluate(predicts,real,params):
    '''
    Here we evaluate the accuracy received by our model on the test set and print it to console.

    :param predicts: Model predictions
    :param real: Test ground truth
    :param params: Default Parameters
    '''

    print("The Test Error is %f" %(1-accuracy_score(real,predicts)))
    return


def validation(params,param_to_validate,possible_values):
    '''
    Hyper parameter validation process, goes over all given values (possible values), finds the best accuracy on the
    validation set, and uses that parameter value to train a model, to evaluate on the test set.

    :param params: Default Parameters
    :param param_to_validate: a hyper parameter which we want to examine
    :param possible_values: the hyper parameter range of values we wants to examine in an iterable object
    :return: Does not return anything
    '''
    train_images, train_labels, test_images, test_labels = load_data(params)

    train_accuracies = []
    val_accuracies = []

    model, history = base_train_model(params,train_images,train_labels)
    best_val_acc = history['val_accuracy'][-1]
    best_value = 0
    i = 0
    for value in possible_values:

        params[param_to_validate] = value
        model, history = train_best_model(params, train_images, train_labels)
        train_accuracies.append(history['accuracy'][-1])
        val_accuracies.append(history['val_accuracy'][-1])
        if val_accuracies[i] >= best_val_acc:
            best_val_acc = val_accuracies[i]
            best_value = value
        i += 1
    plt.plot(possible_values, train_accuracies, label='Training')
    plt.plot(possible_values, val_accuracies, label='Validation')
    plt.xlabel(param_to_validate)
    plt.ylabel("Prediction Accuracy")
    plt.title("Accuracy as a function of %s" % (param_to_validate))
    plt.legend(loc='lower right')
    plt.show()
    print("the best value of parameter %s is %f" %(param_to_validate,best_value))
    if best_value !=0:
        params[param_to_validate] = best_value
        model, history = train_best_model(params, train_images, train_labels)
        display_learning(history)
        pred, pred_prob = test(model, test_images, params)
        evaluate(pred, test_labels, params)


def display_learning(history):
    '''
       Displays the history of the learning process (accuracy & loss as a function of epochs)
       :param history: dictionary containing loss & accuracy vectors for each epoch in the training process.
       :return: Does not return anything
       '''
    acc = history['accuracy']
    val_acc = history['val_accuracy']

    loss = history['loss']
    val_loss = history['val_loss']

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
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

# Main starts here

keras = tf.keras # using tensorflow as beckend for keras
params = GetDefaultParameters() # Getting the default parameters

params['path'] = params['path'] # You can change the path here instead of inside the get function if you prefer.
params['test_indices'] = params['test_indices'] # You can change the indices here instead of inside the get function if you prefer.

tf.random.set_seed(params['seed']) # Setting the random seed for tensorflow
np.random.seed(params['seed']) # Setting the random seed for numpy

if not params['validate']:
    train_images, train_labels, test_images, test_labels = load_data(params) # get processed images split to train and test
    model, history = train_best_model(params,train_images,train_labels) # Train the best model
    #display_learning(history) # Plots history of learning - loss & accuracy as a function of epochs
    pred, pred_prob = test(model,test_images,params) # Get predictions for the test data-set
    precision_recall_graph(pred_prob, test_labels)  # Plot precision recall graph
    #errors(test_images,pred,pred_prob,test_labels) #shows the 5 worst errors (type 1 and type 2)
    evaluate(pred,test_labels,params) # Show error on test
else:
    model = validation(params,"learning_rate" ,(0.005,0.01,0.05,0.1,0.5)) # Hyper validation process
