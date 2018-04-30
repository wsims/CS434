"""
K-Nearest Neighbor Implementation 

Makes predictions for maligant (positiv) and benign (negative) forms of
breast cancer using the K-Nearest Neighbor algorithm. For each possible 
value of K, the training error (measured as the number of mistakes), the
leave-one-out cross-validation error on the training set, the number of 
errors on the provided test data are calculated and plotted as a function
of K.     

This script is used to satisfy part 1 in its entirety for assignment 2.

Usage:
    $ python knn.py

Note: 'knn.py', 'knn_test.csv', and 'knn_train.csv' must all be
in the same folder when running this program. Additionally, the code that
plots the number of errors onto a graph has been commented out incase
matplotlib is not installed. 

"""

import numpy as np
import math
import matplotlib.pyplot as plt

def read_data(file):
    """Opens a .csv file, processes the data into a 2D python list, and
    returns the data.
    
    Inputs:
        file (str): the name of a csv file.
    
    Outputs:
        data (2D list): the data set read from the file.

    """
    x_list = []
    y_list = []
    data = []
    f = open(file, 'r')

    # Read values from file
    for line in f:
        value_list = map(float, line.split(','))
        x = value_list[1:]
        y = value_list[0]
        x_list.append(x)
        y_list.append(y)

    # Normalize features
    for feature in range(30):
        max = 0
        for observation in x_list:
            if max < observation[feature]:
                max = observation[feature]
        for observation in x_list:
            observation[feature] = observation[feature]/max

    # Create data set object
    for i in range(len(x_list)):
        x = np.matrix(x_list[i]).T
        y = y_list[i]
        data.append([x, y])

    return data

def distance(x1, x2):
    return math.sqrt(((x1-x2).T*(x1-x2)).item(0))

def knn(train_data, x, k):
    distance_list = []
    for obs in train_data:
        dist = distance(obs[0], x)
        y = obs[1]
        distance_list.append([dist, y])

    sort_dist_list = sorted(distance_list, key=lambda x: x[0])

    count = 0
    for i in range(k):
        if abs(sort_dist_list[i][1]-1) < 1E-10:
            count += 1

    if count > k/2:
        predict = 1
    else:
        predict = -1

    return predict

def knn_errors(train_data, test_data, k):
    predictions = []
    for obs in test_data:
        predictions.append(knn(train_data, obs[0], k))
 
    errors = 0
    for i in range(len(train_data)):
        if ((train_data[i][1] < 0 and predictions[i] > 0) or (train_data[i][1] > 0 and predictions[i] < 0)):
            errors += 1
    
    return errors

def knn_leave_one(train_data, test_data, k):
    predictions = []
    for i, obs in enumerate(train_data):
        temp = np.copy(obs)
        train_data = np.delete(train_data, i, 0)
        predictions.append(knn(train_data, temp[0], k))
        train_data = np.insert(train_data, i, temp, 0)

    errors = 0
    for i in range(len(train_data)):
        if ((train_data[i][1] < 0 and predictions[i] > 0) or (train_data[i][1] > 0 and predictions[i] < 0)):
            errors += 1
    
    return errors
            
if __name__ == '__main__':
    training_data = read_data('knn_train.csv')
    testing_data = read_data('knn_test.csv')
    
    train_error = []
    leave_one_error = []
    test_data_error = []
    for k in range(1, 51, 2):
        train_error.append(knn_errors(training_data, training_data, k))
        leave_one_error.append(knn_leave_one(training_data, training_data, k))
        test_data_error.append(knn_errors(training_data, testing_data, k))
    
    k = range(1, 51, 2)
    plt.plot(k, train_error, '-b', label='training error')
    plt.plot(k, leave_one_error, '-g', label='leave-one-out error')
    plt.plot(k, test_data_error, '-r', label='errors on test data')
    plt.legend(loc='lower right')
    plt.xlabel('K')
    plt.ylabel('Number of errors')
    plt.title('Number of KNN errors as a function of d')
    plt.savefig("part1.png")
    # print 'Plot saved as "part1.png"
