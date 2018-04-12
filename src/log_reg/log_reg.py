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

def main():
    w = np.matrix([0]*257).T
    while True:
        w, gradient = batch_train(w)
        if np.linalg.norm(gradient) < 100:
            break

    print w

main()
