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
        for i in range(len(self.positive_data)/2):
            choice = random.randint(0, len(self.positive_data)-1)
            bootstrap_set.append(self.positive_data[choice])
        for i in range(len(self.data)):
            choice = random.randint(0, len(self.data)-len(self.positive_data)/2-1)
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

class PerformanceEval(object):

    def __init__(self):
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        self.total = 0

    def add_result(self, prediction, label):
        self.total += 1
        if abs(prediction - 1.0) < 1E-10:
            if abs(label - 1.0) < 1E-10:
                self.TP += 1
            else:
                self.FP += 1
        else:
            if abs(label - 1.0) < 1E-10:
                self.FN += 1
            else:
                self.TN += 1

    def __add__(self, other):
        new = PerformanceEval()
        new.TP = self.TP + other.TP
        new.FN = self.FN + other.FN
        new.FP = self.FP + other.FP
        new.TN = self.TN + other.TN
        new.total = self.total + other.total
        return new

    def accuracy(self):
        return float(self.TP + self.TN)/float(self.total)

    def recall(self):
        return float(self.TP)/float(self.TP + self.FN)

    def precision(self):
        return float(self.TP)/float(self.TP + self.FP)

    def F1(self):
        return float(2*self.TP)/float(2*self.TP + self.FP + self.FN)

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
    eval = PerformanceEval()
    for row in data:
        eval.add_result(forest.predict_binary(row), row[0])
    return eval

def cross_validate(data_list, forest_size):
    for i in range(4):
        data = data_list[i % 4] + data_list[(i + 1) % 4] + data_list[(i + 2) % 4]
        forest = RandomForest(data, forest_size)
        eval = compute_accuracy(forest, data_list[(i + 3) % 4])


if __name__ == "__main__":

    data_list = []

    # Subject A = 1, B = 4, C = 6, D = 9
    window_list, window_label = dp.get_window_data("train_data/Subject_1.csv",
                                                   "train_data/list_1.csv")

    data_list.append(dtree_data_trans(window_list, window_label))

    window_list, window_label = dp.get_window_data("train_data/Subject_4.csv",
                                                   "train_data/list_4.csv")

    data_list.append(dtree_data_trans(window_list, window_label))

    window_list, window_label = dp.get_window_data("train_data/Subject_6.csv",
                                                   "train_data/list_6.csv")

    data_list.append(dtree_data_trans(window_list, window_label))

    window_list, window_label = dp.get_window_data("train_data/Subject_9.csv",
                                                   "train_data/list_9.csv")

    data_list.append(dtree_data_trans(window_list, window_label))

    


    forest = RandomForest(data, 9)
    eval = compute_accuracy(forest, data)
    print("The training accuracy is: %f" % eval.accuracy())
    print("The training precision is: %f" % eval.precision())
    print("The training recall is: %f" % eval.recall())
    print("The training F1 measure is: %f" % eval.F1())

#    d = 10
#    for d in range(10, 20):
#        dtree = dt.build_tree(data, d)
#
#        train_accuracy = dt.compute_accuracy(dtree, data)
#        print("For a tree of depth %d:" % d)
#        print("The training accuracy is: %f" % train_accuracy)
#
