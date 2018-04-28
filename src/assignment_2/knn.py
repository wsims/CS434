import numpy as np
import math
import matplotlib.pyplot as plt

def read_data(file):
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

def knnErrors(train_data, test_data, k):
    predictions = []
    for obs in test_data:
        predictions.append(knn(train_data, obs[0], k))
 
    errors = 0
    for i in range (len(training_data)):
        if ((train_data[i][1] < 0 and predictions[i] > 0) or (train_data[i][1] > 0 and predictions[i] < 0)):
            errors += 1
    
    return errors

if __name__ == '__main__':
    training_data = read_data('knn_train.csv')
    testing_data = read_data('knn_test.csv')
    
    train_error = []
    cross_val_error = []
    test_data_error = []
    for k in range(1, 51, 2):
        train_error.append(knnErrors(training_data, training_data, k))
        test_data_error.append(knnErrors(training_data, testing_data, k))

    runs = range(1, 51, 2)
    plt.plot(runs, train_error, '-b', label='training error')
    plt.plot(runs, test_data_error, '-r', label='errors on test data')
    plt.legend(loc='lower right')
    plt.xlabel('K')
    plt.ylabel('Number of errors')
    plt.title('Numing of KNN errors as a function of d')
    plt.savefig("part1.png")
    # print 'Plot saved as "part1.png"'
