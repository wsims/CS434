import numpy as np
import math
import matplotlib.pyplot as plt

class DecisionTree:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.early_leaf = False
        self.data = data
        pos_count, count = count_data(data)
        if pos_count == 0 or pos_count == count:
            self.early_leaf = True
        if self.early_leaf != True:
            self.feature, self.split, self.info_gain = get_best_split(data)
        self.expect = get_expect(pos_count, count)
        #print("Feature: %d" % self.feature)
        #print("Split: %f" % self.split)
        #print("Expected value: %d" % self.expect)

    def split_data(self):
        data_low = []
        data_high = []
        for row in self.data:
            if row[self.feature] < self.split:
                data_low.append(row)
            else:
                data_high.append(row)
        return data_low, data_high

    def add_layer(self):
        self._add_nodes()

    def _add_nodes(self):
        if self.left == None and self.right == None:
            # We will add two nodes here as long as the data is not a pure set
            if self.early_leaf != True:
                data_low, data_high = self.split_data()
                self.left = DecisionTree(data_low)
                self.right = DecisionTree(data_high)
        else:
            if self.left != None:
                self.left._add_nodes()
            if self.right != None:
                self.right._add_nodes()

    def predict_data(self, row):
        node = self
        while (node.left != None):
            if row[node.feature] < node.split:
                node = node.left
            else:
                node = node.right
        return node.expect

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

    return (opt_feature, opt_split, opt_info_gain)

def count_data(data):
    count = 0
    pos_count = 0
    for row in data:
        if abs(row[0] - 1) < 1E-10:
            pos_count += 1
        count += 1
    return pos_count, count

def get_expect(pos_count, count):
    if float(pos_count)/float(count) >= 0.5:
        prediction = 1
    else:
        prediction = -1
    return prediction

def split_data(data, feature, split):
    data_low = []
    data_high = []
    for row in data:
        if row[feature] < split:
            data_low.append(row)
        else:
            data_how.append(row)
    return data_low, data_high

def build_tree(data, k=0):
    tree = DecisionTree(data)
    for i in range(k):
        tree.add_layer()
    return tree

def compute_accuracy(tree, data):
    correct = 0
    for row in data:
        if abs(tree.predict_data(row) - row[0]) < 1E-10:
            correct += 1
    return float(correct)/float(len(data))

def plot_error(train_error_list, test_error_list):
    d = range(1, len(train_error_list) + 1)
    train_error_list = map(lambda x: 100*x, train_error_list)
    test_error_list = map(lambda x: 100*x, test_error_list)
    plt.plot(d, train_error_list, '-b', label='training error rate')
    plt.plot(d, test_error_list, '-r', label='test error rate')
    plt.legend(loc='lower left')
    plt.xlabel('Decision Tree Depth Limit')
    plt.ylabel('Classification Percent Error Rate')
    plt.title('Training and Testing Error Rates as a Function of Tree Depth Limit')
    plt.savefig("dtree_plot.png")
    print 'Plot saved as "dtree_plot.png"'

if __name__ == '__main__':
    train_data = read_data('knn_train.csv')
    test_data = read_data('knn_test.csv')
    feature, split, info_gain = get_best_split(train_data)
    print("Optimal feature to split over: %d" % feature)
    print("Optimal split: %f" % split)
    print("Optimal info gain: %f" % info_gain)

    train_error_list = []
    test_error_list = []

    for d in range(1, 7):
        dtree = build_tree(train_data, d)

        train_accuracy = compute_accuracy(dtree, train_data)
        test_accuracy = compute_accuracy(dtree, test_data)
        train_error_list.append(1 - train_accuracy)
        test_error_list.append(1 - test_accuracy)
        print("For a tree of depth %d:" % d)
        print("The training error is: %f" % (1 - train_accuracy))
        print("The testing error is: %f" % (1 - test_accuracy))
        print("")

    plot_error(train_error_list, test_error_list)

