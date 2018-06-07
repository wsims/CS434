import numpy as np
import random

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from assignment_2 import decision_tree as dt
import data_process as dp

class RandomForest(object):

    def __init__(self, data, size):
        self.size = size
        self.trees = []
        self.negative_data = None
        self.positive_data = None
        self.data = data

        self.negative_data, self.positive_data = self._split_data(data)
        for i in range(size):
            print "Generating tree %d..." % (i + 1)
            b_set = self._get_bootstrap()
            self.trees.append(dt.build_tree(b_set, 20))

    def _split_data(self, data):
        negative_data = []
        positive_data = []

        for row in data:
            if row[0] == -1:
                negative_data.append(row)
            else:
                positive_data.append(row)

        return negative_data, positive_data

    def _get_bootstrap(self):
        bootstrap_set = []
        for i in range(len(self.data)):
            choice = random.randint(0, len(self.data)-1)
            bootstrap_set.append(self.data[choice])
#        bootstrap_set = self.positive_data[:]
#        for i in range(int(7.0/3.0*len(self.positive_data))):
#            choice = random.randint(0, len(self.negative_data)-1)
#            bootstrap_set.append(self.negative_data[choice])
#
        return bootstrap_set

    def predict_binary(self, observation):
        prediction = -1
        positive_predicts = 0
        for tree in self.trees:
            if abs(tree.predict_data(observation) - 1.0) < 1E-10:
                positive_predicts += 1

        if float(positive_predicts)/float(self.size) >= 0.5:
            prediction = 1
        return prediction

def dtree_data_trans(data_list, label_list):
    full_array = []

    for i, row in enumerate(data_list):
        label_value = 1
        if label_list[i] == 0:
            label_value = -1
        full_array.append([label_value] + row)

    return full_array

def compute_accuracy(forest, data):
    correct = 0
    for row in data:
        if abs(forest.predict_binary(row) - row[0]) < 1E-10:
            correct += 1
    return float(correct)/float(len(data))

if __name__ == "__main__":
    window_list, window_label = dp.get_window_data("train_data/Subject_1.csv",
                                                   "train_data/list_1.csv")

    data = dtree_data_trans(window_list, window_label)

    forest = RandomForest(data, 9)
    accuracy = compute_accuracy(forest, data)
    print("The training accuracy is: %f" % accuracy)

#    d = 10
#    for d in range(10, 20):
#        dtree = dt.build_tree(data, d)
#
#        train_accuracy = dt.compute_accuracy(dtree, data)
#        print("For a tree of depth %d:" % d)
#        print("The training accuracy is: %f" % train_accuracy)
#
