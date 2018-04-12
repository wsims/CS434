import numpy as np
from scipy.stats import logistic
import math

def get_xy(value_list):
    value_list.insert(0, 1)
    length = len(value_list)
    y = value_list[length-1]
    x = np.matrix(value_list[:length-1]).T
    return x, y

def batch_train(w, learning_rate=0.5, file='usps-4-9-train.csv'):
    gradient = np.matrix([0]*257).T
    f = open(file, 'r')

    for line in f:
        value_list = map(float, line.split(','))
        x, y = get_xy(value_list)
        y_hat = logistic.cdf(w.T*x).item(0)
        gradient = gradient + (y_hat - y)*x

    w = w - learning_rate*gradient
    return w, gradient

def predict(w, x, prob_threshold=0.75):
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

    print "Total trials: %d" % count
    print "Total correct: %d" % correct
    return float(correct)/float(count)

def main():
    w = np.matrix([0]*257).T
    while True:
        w, gradient = batch_train(w)
        if np.linalg.norm(gradient) < 1000:
            break

    accuracy = test_accuracy(w, 'usps-4-9-train.csv')
    print("Training data classifier accuracy: %f" % accuracy)
    accuracy = test_accuracy(w, 'usps-4-9-test.csv')
    print("Test data classifier accuracy: %f" % accuracy)

    # print w

main()
