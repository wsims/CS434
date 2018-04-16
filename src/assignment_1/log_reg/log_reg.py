"""
Usage: python log_reg.py

Uses a learning rate of 6 and a gradient norm stopping condition
of 5000 to train a binary logistic regression classifier to identify
handwritten 4s vs handwritten 9s.  Creates a plot of training
accuracy and testing accuracy against the number of gradient 
descents performed.  The plot is saved as "log_reg.png".

This script is used to satisfy question 1 for part 2 of assignment 1.

"""

import numpy as np
from scipy.stats import logistic
# import matplotlib.pyplot as plt
import math

def get_xy(value_list):
    """Returns x vector and y value from value list in file.
    
    Inputs:
        value_list -- a python list of data taken from a single row in
            the training file.

    Outputs:
        x -- a numpy matrix object in the form of a column vector
        y -- the last value from the 'value_list' input
    
    """
    value_list.insert(0, 1)
    length = len(value_list)
    y = value_list[length-1]
    x = np.matrix(value_list[:length-1]).T
    return x, y

def batch_train(w, learning_rate=6, file='usps-4-9-train.csv'):
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
    gradient = np.matrix([0]*257).T
    f = open(file, 'r')

    for line in f:
        value_list = map(float, line.split(','))
        x, y = get_xy(value_list)
        y_hat = logistic.cdf(w.T*x).item(0)
        gradient = gradient + (y_hat - y)*x

    w = w - learning_rate*gradient
    return w, gradient

def predict(w, x, prob_threshold=0.5):
    """Returns 0 if system predicts the image to be a 4.
    1 if the system predicts image to be a 9.

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
    if prob > prob_threshold:
        return_value = 1
    return return_value

def test_accuracy(w, file):
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
    f = open(file, 'r')
    count = 0
    correct = 0

    for line in f:
        count += 1
        value_list = map(float, line.split(','))
        x, y = get_xy(value_list)
        if predict(w, x) == int(y):
            correct += 1

    return float(correct)/float(count)

if __name__ == '__main__':
    count = 0
    training_acc = []
    testing_acc = []

    w = np.matrix([0]*257).T
    while True:
        w, gradient = batch_train(w)

        count += 1
        training_acc.append(test_accuracy(w, 'usps-4-9-train.csv'))
        testing_acc.append(test_accuracy(w, 'usps-4-9-test.csv'))
        if np.linalg.norm(gradient) < 5000:
            break

    runs = range(1, count + 1)
#     plt.plot(runs, training_acc, '-b', label='training data')
#     plt.plot(runs, testing_acc, '-r', label='testing data')
#     plt.legend(loc='lower right')
#     plt.xlabel('Number of Gradient Descent Iterations')
#     plt.ylabel('Accuracy')
#     plt.title('Model Accuracy')
#     plt.savefig("log_reg.png")
#     print 'Plot saved as "log_reg.png"'
# 
    accuracy = test_accuracy(w, 'usps-4-9-train.csv')
    print("Training data classifier accuracy: %f" % accuracy)
    accuracy = test_accuracy(w, 'usps-4-9-test.csv')
    print("Test data classifier accuracy: %f" % accuracy)

