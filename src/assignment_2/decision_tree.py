import numpy as np
import math

def read_data(file):
    data = []
    f = open(file, 'r')

    for line in f:
        value_list = map(float, line.split(','))
        data.append(value_list)

    return data

def sort_by_feature(data, feature):
    return sorted(data, key=lambda x: x[feature])

def entropy(total_count, positive_count):
    # Probabilities
    p_pos = float(positive_count)/float(total_count)
    p_neg = 1 - p_pos
   
    if total_count == positive_count or positive_count == 0:
        H_s = 0
    else:
        H_s = - p_pos * math.log(p_pos, 2) - p_neg * math.log(p_neg, 2)
    return H_s

def entropy_of_split(sort_data, feature, split, total_positives):
    positives = 0
    count = 0
    while sort_data[count][feature] < split:
        if abs(sort_data[count][0] - 1) < 1E-10:
            positives += 1
        count += 1
    H_s1 = entropy(count, positives)
    H_s2 = entropy(len(sort_data)-count, total_positives-positives)
    H_split = float(count) / len(sort_data) * H_s1
    H_split += (1 - float(count) / len(sort_data)) * H_s2
    return H_split

def find_bin_split(data, feature):
    """Feature is a value between 1 and 30 inclusive"""
    split_list = []
    sort_data = sort_by_feature(data, feature)
    for i in range(1, len(data)):
        if abs(sort_data[i][0] - sort_data[i-1][0]) > 1E-10:
            split_list.append((sort_data[i][feature] + sort_data[i-1][feature])/2)
    
    # Find initial Entropy
    pos_count = 0
    for row in sort_data:
        if abs(row[0]-1) < 1E-10:
            pos_count += 1
    H_initial = entropy(len(sort_data), pos_count)

    # Find best split and highest info gain
    info_gain = 0
    best_split = 0
    for split in split_list:
        H_split = entropy_of_split(sort_data, feature, split, pos_count)
        if H_initial - H_split > info_gain:
            info_gain = H_initial - H_split
            best_split = split

    return (info_gain, best_split)

def get_best_split(data):
    opt_info_gain = 0
    opt_split = 0
    opt_feature = 0
    for feature in range(1, 31):
        info_gain, split = find_bin_split(data, feature)
        if info_gain > opt_info_gain:
            opt_feature = feature
            opt_split = split
            opt_info_gain = info_gain

    print("Optimal feature to split over: %d" % opt_feature)
    print("Optimal split: %f" % opt_split)
    print("Optimal info gain: %f" % opt_info_gain)
    return (opt_feature, opt_split, opt_info_gain)

def compute_error_rate(train_data, test_data, feature, split):
    sort_train_data = sort_by_feature(train_data, feature)
    count = 0
    pos_count = 0
    while sort_train_data[count][feature] < split:
        if abs(sort_train_data[count][feature] - 1) < 1E-10:
            pos_count += 1
        count += 1

    low_pred = -1
    if float(pos_count)/float(count) >= 0.5:
        low_pred = 1
    
    high_count = 0
    pos_count = 0
    while count < len(sort_train_data):
        if abs(sort_train_data[count][feature] - 1) < 1E-10:
            pos_count += 1
        count += 1
        high_count += 1

    high_pred = -1
    if float(pos_count)/float(high_count) >= 0.5:
        high_pred = 1
    
    # Use prediction values to predict test values
    correct = 0
    for row in test_data:
        if row[feature] < split and abs(low_pred - row[0]) < 1E-10:
            correct += 1
        elif row[feature] >= split and abs(high_pred - row[0]) < 1E-10:
                correct += 1
    
    return 1-float(correct)/float(len(test_data))


if __name__ == '__main__':
    train_data = read_data('knn_train.csv')
    test_data = read_data('knn_test.csv')
    feature, split, info_gain = get_best_split(train_data)
    train_error = compute_error_rate(train_data, train_data, feature, split)
    test_error = compute_error_rate(train_data, test_data, feature, split)
    print("The training error is: %f" % train_error)
    print("The testing error is: %f" % test_error)
    
