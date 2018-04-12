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

def batch_train(w, learning_rate=6, file='usps-4-9-train.csv', lamb=1):
    gradient = np.matrix([0]*257).T
    f = open(file, 'r')

    for line in f:
        value_list = map(float, line.split(','))
        x, y = get_xy(value_list)
        y_hat = logistic.cdf(w.T*x).item(0)
        gradient = gradient + (y_hat - y)*x

    gradient = gradient + lamb*w
    w = w - learning_rate*gradient
    return w, gradient

def predict(w, x, prob_threshold=0.5):
    """Returns 0 if system predicts the image to be a 4.
        1 if the system predicts image to be a 9.
    """
    return_value = 0
    prob = logistic.cdf(w.T*x).item(0)
    if prob > prob_threshold:
        return_value = 1
    return return_value

def test_accuracy(w, file):
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
    w = np.matrix([0]*257).T
    while True:
        w, gradient = batch_train(w, lamb)
        if np.linalg.norm(gradient) < 5000:
            break

    print("Lambda test %.8f" % lamb)
    accuracy = test_accuracy(w, 'usps-4-9-train.csv')
    print("Training data classifier accuracy: %f" % accuracy)
    accuracy = test_accuracy(w, 'usps-4-9-test.csv')
    print("Test data classifier accuracy: %f" % accuracy)
    print ""

if __name__ == '__main__':
    train_and_test(lamb=10.0**-8)
    train_and_test(lamb=10.0**-7)
    train_and_test(lamb=10.0**-6)
    train_and_test(lamb=10.0**-5)
    train_and_test(lamb=10.0**-4)
    train_and_test(lamb=10.0**-3)
    train_and_test(lamb=10.0**-2)
