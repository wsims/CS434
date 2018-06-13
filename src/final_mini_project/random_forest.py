import numpy as np
import random

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from assignment_2 import decision_tree as dt
import data_process as dp
import performance as perf

CROSS_VALIDATE = 0
CLASSIFY_TEST_DATA = 1

#MODE = CROSS_VALIDATE
MODE = CLASSIFY_TEST_DATA

class RandomForest(object):

    def __init__(self, data, size):
        self.size = size
        self.trees = []
        self.negative_data = None
        self.positive_data = None
        self.data = data

        self.negative_data, self.positive_data = self._split_data(data)
        for i in range(size):
            #print "Generating tree %d..." % (i + 1)
            b_set = self._get_bootstrap()
            self.trees.append(dt.build_tree(b_set, 50))

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
        for i in range(int(len(self.positive_data))):
            choice = random.randint(0, len(self.positive_data)-1)
            bootstrap_set.append(self.positive_data[choice])
        for i in range(10*int(len(self.positive_data))):
            choice = random.randint(0, len(self.data)-1)
            bootstrap_set.append(self.data[choice])

        return bootstrap_set

    def predict(self, observation):
        prediction = -1
        positive_predicts = 0
        for tree in self.trees:
            if abs(tree.predict_data(observation) - 1.0) < 1E-10:
                positive_predicts += 1

        if float(positive_predicts)/float(self.size) >= 0.5:
            prediction = 1
        return float(positive_predicts)/float(self.size)

def float_to_binary_cross(pred):
    value = -1
    if pred >= 0.5:
        value = 1
    return value

def float_to_binary(pred):
    value = 0
    if pred >= 0.5:
        value = 1
    return value

def dtree_test_data_trans(data_list):
    full_array = []

    for row in data_list:
        full_array.append([0] + row) # Add a placeholder column where labels would go

    return full_array

def dtree_data_trans(data_list, label_list):
    full_array = []

    for i, row in enumerate(data_list):
        label_value = 1
        if label_list[i] == 0:
            label_value = -1
        full_array.append([label_value] + row)

    return full_array

def compute_accuracy(forest, data):
    eval = perf.PerformanceEval()
    for row in data:
        eval.add_result(float_to_binary_cross(forest.predict(row)), row[0])
    return eval

def cross_validate(data_list, forest_size):
    eval = None
    for i in range(4):
        print "Validating set %d..." % i
        data = data_list[i % 4] + data_list[(i + 1) % 4] + data_list[(i + 2) % 4]
        forest = RandomForest(data, forest_size)
        if i == 0:
            eval = compute_accuracy(forest, data_list[(i + 3) % 4])
        else:
            eval = eval + compute_accuracy(forest, data_list[(i + 3) % 4])

    return eval

def classify(train_data, test_data, n, file="test.csv"):
    print "Building trees... (may take awhile)"
    forest = RandomForest(train_data, n)

    f = open(file, "w")

    print "Classifying Test Data..."
    pos_count = 0
    for row in test_data:
        pred = forest.predict(row)
        if pred >= 0.5:
            pos_count += 1
        f.write("%f,%d\n" % (pred, float_to_binary(pred)))

    f.close()
    print "Classification file saved as '" + file + "'"
    print "Found %d positives out of %d data points!" % (pos_count, len(test_data))


def split_data(data):
    negative_data = []
    positive_data = []

    for row in data:
        if row[0] == -1:
            negative_data.append(row)
        else:
            positive_data.append(row)

    return negative_data, positive_data


def get_cross_sets(data):
    random.shuffle(data)
    n_data, p_data = split_data(data)
    data_list = []
    data_list.append(n_data[:len(n_data)/4] + p_data[:len(p_data)/4])
    data_list.append(n_data[len(n_data)/4:len(n_data)/2] + p_data[len(p_data)/4:len(p_data)/2])
    data_list.append(n_data[len(n_data)/2:3*len(n_data)/4] + p_data[len(p_data)/2:3*len(p_data)/4])
    data_list.append(n_data[3*len(n_data)/4:] + p_data[3*len(p_data)/4:])

    return data_list

if __name__ == "__main__":

    if MODE == CROSS_VALIDATE:
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

        individual_data = []

        window_list, window_label = dp.get_window_data("train_data/Subject_2_part1.csv",
                                                       "train_data/list2_part1.csv")

        individual_data = dtree_data_trans(window_list, window_label)

        indie_data_list = get_cross_sets(individual_data)

        # Cross validation
        domain = range(71, 72, 2)
        f1_list = []

        for i in range(71, 72, 2):
            print "Running test for forest size %d" % i
            f1_list.append(cross_validate(indie_data_list, i).F1())
            print ""

        best_index = f1_list.index(max(f1_list))

        print "Highest F1 score found when forest size was %d" % domain[best_index]
        print "F1 score at peak: %f" % f1_list[best_index]

        print f1_list

    elif MODE == CLASSIFY_TEST_DATA:
        train_data = []
        window_list, window_label = dp.get_window_data("train_data/Subject_1.csv",
                                                       "train_data/list_1.csv")

        train_data += dtree_data_trans(window_list, window_label)

        window_list, window_label = dp.get_window_data("train_data/Subject_4.csv",
                                                       "train_data/list_4.csv")

        train_data += dtree_data_trans(window_list, window_label)

        window_list, window_label = dp.get_window_data("train_data/Subject_6.csv",
                                                       "train_data/list_6.csv")

        train_data += dtree_data_trans(window_list, window_label)

        window_list, window_label = dp.get_window_data("train_data/Subject_9.csv",
                                                       "train_data/list_9.csv")

        train_data += dtree_data_trans(window_list, window_label)

        test_data = dp.get_test_data("test_data/general_test_instances.csv")
        test_data = dtree_test_data_trans(test_data)

        classify(train_data, test_data, 71, file="general_pred2.csv")

        # Individual 1
        train_data = []
        window_list, window_label = dp.get_window_data("train_data/Subject_2_part1.csv",
                                                       "train_data/list2_part1.csv")

        train_data = dtree_data_trans(window_list, window_label)

        test_data = dp.get_test_data("test_data/subject2_instances.csv")
        test_data = dtree_test_data_trans(test_data)

        classify(train_data, test_data, 71, file="individual1_pred2.csv")

        # Individual 2
        train_data = []
        window_list, window_label = dp.get_window_data("train_data/Subject_7_part1.csv",
                                                       "train_data/list_7_part1.csv")

        train_data = dtree_data_trans(window_list, window_label)

        test_data = dp.get_test_data("test_data/subject7_instances.csv")
        test_data = dtree_test_data_trans(test_data)

        classify(train_data, test_data, 71, file="individual2_pred2.csv")

    #eval = cross_validate(data_list, 45)
    #print("The cross validation accuracy is: %f" % eval.accuracy())
    #print("The cross validation precision is: %f" % eval.precision())
    #print("The cross validation recall is: %f" % eval.recall())
    #print("The cross validation F1 measure is: %f" % eval.F1())

    #eval = cross_validate(indie_data_list, 45)
    #print("The indie cross validation accuracy is: %f" % eval.accuracy())
    #print("The indie cross validation precision is: %f" % eval.precision())
    #print("The indie cross validation recall is: %f" % eval.recall())
    #print("The indie cross validation F1 measure is: %f" % eval.F1())

