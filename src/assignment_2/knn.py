import numpy as np
import math

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

def knn(train_data, x, k=1):
    distance_list = []
    for obs in train_data:
        dist = distance(obs[0], x)
        y = obs[1]
        distance_list.append([dist, y])

    sort_dist_list = sorted(distance_list, key=lambda x: x[0])

    count = 0
    for i in range(k):
        if sort_dist_list[i][1] == 1:
            count += 1

    if count > k/2:
        predict = 1
    else:
        predict = -1

    return predict

if __name__ == '__main__':
    training_data = read_data('knn_train.csv')
    testing_data = read_data('knn_test.csv')
    

