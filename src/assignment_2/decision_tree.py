"""
Learned Decision Tree Implementation

Generates learned decision trees of various maximum depths to classify
malignant (positive) and benign (negative) forms of breast cancer based on
30 separate features.  Trees are generated using the top-down greedy
induction algorithm where the information gain criterion is the metric
used to determine tree construction.  Calculates error rates for various 
maximum depths for both the training and testing data sets and plots the
results.  The generated plot is stored as 'dtree_plot.png'.  The training
data used is stored in the file 'knn_train.csv' and the testing data used
is stored in the file 'knn_test.csv'.

This script is used to satisfy part 2 in its entirety for assignment 2.

Usage:
    $ python decision_tree.py

Note: 'decision_tree.py', 'knn_test.csv', and 'knn_train.csv' must all be
    in the same folder when running this program.

"""
import numpy as np
import math
import matplotlib.pyplot as plt

class DecisionTree:
    """Learned decision tree object"""

    def __init__(self, data):
        self.left = None
        self.right = None
        self.early_leaf = False
        self.data = data
        pos_count, count = count_data(data)
        if pos_count == 0 or pos_count == count:
            # Mark node as early_leaf if data set is of a single classification
            # (e.g. 100 observations, 100 positive classifications)
            self.early_leaf = True
        if self.early_leaf != True:
            self.feature, self.split, self.info_gain = get_best_split(data)
        self.expect = get_expect(pos_count, count)
        #print("Feature: %d" % self.feature)
        #print("Split: %f" % self.split)
        #print("Expected value: %d" % self.expect)

    def split_data(self):
        """Divides the instance node's data set into two smaller sets 
        based on the feature and split value determined to maximize information
        gain.

        Outputs:
            data_low (2D list): The segment of the original data set whose optimal
                feature values are less than the optimal split value.
            data_high (2D list): The segment of the original data set whose optimal
                feature values are greater than or equal to the optimal split value.

        """
        data_low = []
        data_high = []
        for row in self.data:
            if row[self.feature] < self.split:
                data_low.append(row)
            else:
                data_high.append(row)
        return data_low, data_high

    def add_layer(self):
        """Generates two new nodes at every node where information gain can
        be increased by implementing a decision rule.

        """
        self._add_nodes()

    def _add_nodes(self):
        """Recursive function used to implement 'add_layer' function"""
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

    def predict_data(self, observation):
        """Predicts breast cancer classification using decision tree.
        
        Inputs:
            observation (list): Python list of length 31 where the first value
                is the true classification, and values 2-31 are the feature values.

        Outputs:
            node.expect (int): Expected value.  Either 1 (malignant) or -1 (benign).

        """
        node = self
        while (node.left != None):
            if observation[node.feature] < node.split:
                node = node.left
            else:
                node = node.right
        return node.expect

def read_data(file):
    """Opens a .csv file, processes the data into an 2D python list, and returns
    the data.
    
    Inputs:
        file (str): the name of a csv file.

    Outputs:
        data (2D list): the data set read from the file.

    """
    data = []
    f = open(file, 'r')

    for line in f:
        value_list = map(float, line.split(','))
        data.append(value_list)

    return data

def sort_by_feature(data, feature):
    """Sorts the data points in a data set by the value of one of their features.

    Inputs:
        data (2D list): An N x 31 matrix of data.
        feature (int): An index between 1 and 30 (inclusive) corresponding to the
            feature we wish to sort the data by.

    Outputs:
        sorted_data (2D list): An N x 31 matrix of data sorted by feature value
            where lower index data points have lower values for the sorting
            feature.

    """
    return sorted(data, key=lambda x: x[feature])

def entropy(total_count, positive_count):
    """Calculates the entropy for a set, or subset, of data using the total number
    of observations in the set and the total number of positive classifications
    in the set.

    Inputs:
        total_count (int): Total number of observations.
        positive_count (int): Total number of positively classified observations.

    Outputs:
        H_s (float): Entropy of the data set.

    """
    # Probabilities
    p_pos = float(positive_count)/float(total_count)
    p_neg = 1 - p_pos
   
    if total_count == positive_count or positive_count == 0:
        # Necessary edge case since log(0) is undefined
        H_s = 0
    else:
        H_s = - p_pos * math.log(p_pos, 2) - p_neg * math.log(p_neg, 2)
    return H_s

def entropy_of_split(sort_data, feature, split, total_positives):
    """Calculates the total entropy for a data set that has been segmented into
    two smaller data sets.  The value returned from this function, when subtracted
    from the entropy of the whole data set, gives the information gain of a feature
    split.

    Inputs:
        sort_data (2D list): data which has been sorted according to a specific
            feature.
        feature (int): the index which 'sort_data' was sorted by.  Value must
            be between 1 and 30 (inclusive).
        split (float): the feature threshold used to divide the data set.
        total_positives (int): Total number of positive classifications in the
            data set.

    Outputs:
        H_split (float): total entropy of the split data sets.

    """
    positives = 0
    count = 0

    # Count the number of observations with feature values less than split.
    # Also count the number of positive observations w/feature values less than
    # the split value.
    while sort_data[count][feature] < split:
        if abs(sort_data[count][0] - 1) < 1E-10:
            positives += 1
        count += 1

    # Entropy for set with feature values below threshold
    H_s1 = entropy(count, positives)

    # Entropy for set with feature values above threshold
    H_s2 = entropy(len(sort_data)-count, total_positives-positives)

    # Multiply entropy for individual sets by probability of being moved to that
    # set and sum
    H_split = float(count) / len(sort_data) * H_s1
    H_split += (1 - float(count) / len(sort_data)) * H_s2
    return H_split

def find_bin_split(data, feature):
    """Finds the best threshold to divide a data set according to a specific
    feature in order to maximize information gain.

    Inputs:
        data (2D list): An N x 31 matrix of data.
        feature (int): An index between 1 and 30 (inclusive).

    Outputs:
        info_gain (float): Amount of information gained by splitting the data.
            Value is between 0 and 1.
        best_split (float): The optimal threshold value to divide the data set
            according to the specific input feature.

    """
    split_list = []
    sort_data = sort_by_feature(data, feature)

    # Construct list of possible splits.
    # Optimal splits only occur inbetween data points with different classifications
    for i in range(1, len(data)):
        # If classifications don't match, sort_data[i][0] - sort_data[i-1][0] will be 2.0
        # Otherwise, the value is 0.0
        if abs(sort_data[i][0] - sort_data[i-1][0]) > 1:
            split_list.append((sort_data[i][feature] + sort_data[i-1][feature])/2)

    # Find initial Entropy
    pos_count, count = count_data(sort_data)
    H_initial = entropy(count, pos_count)

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
    """Determines the optimal feature and feature threshold to divide a data set
    on to maximize information gain.

    Inputs:
        data (2D list): An N x 31 matrix of data.

    Outputs:
        opt_feature (int): Optimal feature index to segment the data by.
        opt_split (float): Optimal feature threshold to segment the data by.
        opt_info_gain (float): Amount of information gained by segmenting the
            data according to the feature and threshold value.

    """
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
    """Counts the number of total observations and positively classified
    observations in a data set.

    Inputs:
        data (2D list): An N x 31 matrix of data.

    Outputs:
        pos_count (int): Number of positively classified observations.
        count (int): Number of total observations.

    """
    count = 0
    pos_count = 0
    for row in data:
        if abs(row[0] - 1) < 1E-10:
            pos_count += 1
        count += 1
    return pos_count, count

def get_expect(pos_count, count):
    """Returns the expected value for a data set based on the classification 
    majority.

    Inputs:
        pos_count (int): Number of positively classified observations.
        count (int): Number of total observations.

    Outputs:
        prediction (int): Either -1 or 1.

    """
    if float(pos_count)/float(count) >= 0.5:
        prediction = 1
    else:
        prediction = -1
    return prediction

def split_data(data, feature, split):
    """Divides a data set into two smaller sets based on a specific feature
    and a threshold for that feature.

    Inputs:
        data (2D list): An N x 31 matrix of data.
        feature (int): An index between 1 and 30 (inclusive) corresponding to
            a specific feature in the data set.
        split (float): The threshold used to divide the data set according to
            a specific feature.

    Outputs:
        data_low (2D list): The set of data with feature value less than the
            threshold.
        data_high (2D list): The set of data with feature value greater than
            the threshold.

    """
    data_low = []
    data_high = []
    for row in data:
        if row[feature] < split:
            data_low.append(row)
        else:
            data_high.append(row)
    return data_low, data_high

def build_tree(data, d=0):
    """Construct a decision tree with maximum depth d

    Inputs:
        data (2D list): An N x 31 matrix of data.
        d (int): The maximum depth limit for the decision tree.

    Outputs:
        tree (DecisionTree object): Learned decision tree which can be
            used to classify data.

    """
    tree = DecisionTree(data)
    for i in range(d):
        tree.add_layer()
    return tree

def compute_accuracy(tree, data):
    """Calculate the accuracy of the decision tree classifier for a specific
        data set.

    Inputs:
        tree (DecisionTree object): The classifier.
        data (2D list): An N x 31 matrix of data.

    Outputs:
        accuracy (float): Value between 0 and 1 corresponding to the percentage
            of observations correctly classified.

    """
    correct = 0
    for row in data:
        if abs(tree.predict_data(row) - row[0]) < 1E-10:
            correct += 1
    return float(correct)/float(len(data))

def plot_error(train_error_list, test_error_list):
    """Produce a plot for error rates on the training and testing data sets"""
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
    print("")
    
    train_low, train_high = split_data(train_data, feature, split)
    pos_count, count = count_data(train_data)
    print("Tree root: %d positives and %d negatives" % (pos_count, count - pos_count))
    pos_count, count = count_data(train_low)
    print("low leaf: %d positives and %d negatives" % (pos_count, count - pos_count))
    pos_count, count = count_data(train_high)
    print("high leaf: %d positives and %d negatives\n" % (pos_count, count - pos_count))

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

