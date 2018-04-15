"""
Usage: python log_reg_regular.py

Uses a learning rate of 0.05 and a gradient norm stopping condition
of 50000 to train a binary logistic regression classifier to identify
handwritten 4s vs handwritten 9s.  Utilizes an L2 regularization term.
Prints training accuracy and testing accuracy for different values of
lambda in the L2 regularization term from 10**-7 to 10**-2.

This script is used to satisfy questions 2 and3 for part 2 of assignment 1.

"""
import numpy as np
from scipy.stats import logistic
import matplotlib.pyplot as plt
import math

def get_xy(value_list):
    """Returns x vector and y value from value list in file"""
    value_list.insert(0, 1)
    length = len(value_list)
    y = value_list[length-1]
    x = np.matrix(value_list[:length-1]).T
    return x, y

def batch_train(w, learning_rate=0.05, file='usps-4-9-train.csv', lamb=1):
    """Performs one iteration of gradient descent with a regularization
    term using the full training data set.

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

    gradient = gradient + lamb*w
    w = w - learning_rate*(gradient)
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
        file -- the name of a csv file containing the trend data to be used.

    Outputs:
        accuracy -- a floating point value between 0 and 1 indiciating what
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

def train_and_test(lamb=1):
    """Trains the binary logistic regression classifier using
    the input lambda value then tests model using the training
    data and testing data and prints the result.

    Inputs:
        lamb -- a floating point value.  The L2 regularization term 
            lambda value

    """
    w = np.matrix([0]*257).T
    while True:
        w, gradient = batch_train(w, lamb)
        if np.linalg.norm(gradient) < 50000:
            break

    print("Lambda test %.8f" % lamb)
    accuracy = test_accuracy(w, 'usps-4-9-train.csv')
    print("Training data classifier accuracy: %f" % accuracy)
    accuracy = test_accuracy(w, 'usps-4-9-test.csv')
    print("Test data classifier accuracy: %f" % accuracy)
    print ""

if __name__ == '__main__':
    train_and_test(lamb=10.0**-7)
    train_and_test(lamb=10.0**-6)
    train_and_test(lamb=10.0**-5)
    train_and_test(lamb=10.0**-4)
    train_and_test(lamb=10.0**-3)
    train_and_test(lamb=10.0**-2)
