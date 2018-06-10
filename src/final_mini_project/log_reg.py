import numpy as np
import random
import math
from scipy.stats import logistic
from imblearn.over_sampling import SMOTE

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import data_process as dp

def get_xy(file, list_file):
    """Returns x vector and y value from value list in file.
    
    Inputs:
        value_list -- a python list of data taken from a single row in
            the training file.
    Outputs:
        x -- a numpy matrix object in the form of a column vector
        y -- the last value from the 'value_list' input
    
    """
    x, y = dp.get_window_data(file, list_file)
    x_resampled, y_resampled = SMOTE().fit_sample(x, y)    

    return x_resampled, y_resampled

def batch_train(w, window_list, window_label, learning_rate=6):
    """Performs one iteration of gradient descent using the full
    training data set.
    Inputs:
        w -- a numpy matrix object in the form of a column vector
        learning_rate -- a floating point value used to control
            the gradient descent rate
        file -- the name of a csv file containg the training data to be used
    Outputs:
        w -- a numpy matrix object in the form of a column vector.  This is
            the prediction vector improved by gradient descent.
        gradient -- a numpy matrix object in the form of a column vector.
            This is the gradient used to improve the prediction vector.
    """
    gradient = np.matrix([0]*49).T
    #window_list, window_label = get_xy(file, list_file)

    count = 0
    for line in window_list:
        y = window_label[count]
        count += 1
        x = np.matrix(line).T
        y_hat = logistic.cdf(w.T*x).item(0)
        gradient = gradient + (y_hat - y)*x

    w = w - learning_rate*gradient
    return w, gradient

def predict(w, x, prob_threshold=0.5):
    """Returns 0 if system predicts no hypo event
    1 if the system predicts a hypo event in the next 30 minutes.
    Inputs:
        w -- a numpy matrix object in the form of a column vector.  This
            is the weight vector used to make predictions.
        x -- a numpy matrix object in the form of a column vector.  This
            contains all the feature data for one observation.
        prob_threshold -- a floating point value.  This sets the probability
            threshold which must be achieved for a positive result (i.e. 1)
            to be returned.
    Outputs:
        return_value -- an integer value that is either 0 or 1, depending on what
            the model predicts.
    """
    return_value = 0
    prob = logistic.cdf(w.T*x).item(0)
    print prob
    if prob > prob_threshold:
        return_value = 1
    return return_value

def test_accuracy(w, file, list_file):
    """Tests the model accuracy over an entire data set and returns
    the accuracy.
    Inputs:
        w -- a numpy matrix object in the form of a column vector.  This
            is the weight vector used to make predictions.
        file -- the name of a csv file containg the training data to be used.
    Outputs:
        accuracy -- a floating point value between 0 and 1 indicating what
            percentage of observations were accurately predicted.
    """
    
    window_list, window_label = dp.get_window_data(file, list_file)
                                                 
    count = 0
    correct = 0

    for line in window_list:
        y = window_label[count]
        count += 1
        x = np.matrix(line).T
        if predict(w, x) == int(y):
            correct += 1

    return float(correct)/float(count)

if __name__ == "__main__":
    count = 0
    training_acc = []
    testing_acc = []   
    x, y = get_xy('train_data/Subject_1.csv', 'train_data/list_1.csv')
    w = np.matrix([0]*49).T
    while True:
        w, gradient = batch_train(w, x, y)

        count += 1
        #training_acc.append(test_accuracy(w, 'train_data/Subject_1.csv', 'train_data/list_1.csv'))
        print test_accuracy(w, 'train_data/Subject_1.csv', 'train_data/list_1.csv')
        if np.linalg.norm(gradient) < 10000:
            break
    
    accuracy = test_accuracy(w, 'train_data/Subject_1.csv', 'train_data/list_1.csv')
    print("Training data classifier accuracy: %f" % accuracy)
