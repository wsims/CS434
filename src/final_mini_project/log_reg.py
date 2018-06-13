import numpy as np
import random
import math
from scipy.stats import logistic
from imblearn.over_sampling import SMOTE

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import data_process as dp

class PerformanceEval(object):

    def __init__(self):
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        self.total = 0

    def add_result(self, prediction, label):
        self.total += 1
        if prediction == 1:
            if label == 1:
                self.TP += 1
            else:
                self.FP += 1
        else:
            if label == 1:
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
        if self.FN + self.FN == 0:
            return -1
        return float(self.TP)/float(self.TP + self.FN)

    def precision(self):
        if self.TP + self.FP == 0:
            return -1
        return float(self.TP)/float(self.TP + self.FP)

    def F1(self):
        return float(2*self.TP)/float(2*self.TP + self.FP + self.FN)

def batch_train(w, window_list, window_label, learning_rate=6):
    """Performs one iteration of gradient descent using the full
    training data set.
    Inputs:
        w -- a numpy matrix object in the form of a column vector
        learning_rate -- a floating point value used to control
            the gradient descent rate
        file -- the name of a csv file containg the training data to be used
    Outputs:
        w -- a numpy matrix object in the form of a column vector.  This is
            the prediction vector improved by gradient descent.
        gradient -- a numpy matrix object in the form of a column vector.
            This is the gradient used to improve the prediction vector.
    """
    gradient = np.matrix([0]*49).T
    #window_list, window_label = get_xy(file, list_file)

    count = 0
    for line in window_list:
        y = window_label[count]
        count += 1
        x = np.matrix(line).T
        y_hat = logistic.cdf(w.T*x).item(0)
        gradient = gradient + (y_hat - y)*x

    w = w - learning_rate*gradient
    return w, gradient

def predict(w, x, prob_threshold=0.25):
    """Returns 0 if system predicts no hypo event
    1 if the system predicts a hypo event in the next 30 minutes.
    Inputs:
        w -- a numpy matrix object in the form of a column vector.  This
            is the weight vector used to make predictions.
        x -- a numpy matrix object in the form of a column vector.  This
            contains all the feature data for one observation.
        prob_threshold -- a floating point value.  This sets the probability
            threshold which must be achieved for a positive result (i.e. 1)
            to be returned.
    Outputs:
        return_value -- an integer value that is either 0 or 1, depending on what
            the model predicts.
    """
    return_value = 0
    prob = logistic.cdf(w.T*x).item(0)
    if prob > prob_threshold:
        return_value = 1
    return return_value

def predict_prob(w, x, prob_threshold=0.25):
    """Returns 0 if system predicts no hypo event
    1 if the system predicts a hypo event in the next 30 minutes.
    Inputs:
        w -- a numpy matrix object in the form of a column vector.  This
            is the weight vector used to make predictions.
        x -- a numpy matrix object in the form of a column vector.  This
            contains all the feature data for one observation.
        prob_threshold -- a floating point value.  This sets the probability
            threshold which must be achieved for a positive result (i.e. 1)
            to be returned.
    Outputs:
        return_value -- an integer value that is either 0 or 1, depending on what
            the model predicts and the probably of a positive classification.
    """
    return_value = 0
    prob = logistic.cdf(w.T*x).item(0)
    if prob > prob_threshold:
        return_value = 1
    return return_value, prob    

def test_accuracy(w, file, list_file):
    """Tests the model accuracy over an entire data set and returns
    the accuracy.
    Inputs:
        w -- a numpy matrix object in the form of a column vector.  This
            is the weight vector used to make predictions.
        file -- the name of a csv file containg the training data to be used.
    Outputs:
        accuracy -- a floating point value between 0 and 1 indicating what
            percentage of observations were accurately predicted.
    """
    
    #window_list, window_label = dp.get_window_data(file, list_file)
                                                 
    count = 0
    correct = 0
    eval = PerformanceEval()
    for line in window_list:
        y = window_label[count]
        count += 1
        x = np.matrix(line).T
        eval.add_result(predict(w, x), int(y))
        #if predict(w, x) == int(y):
            #correct += 1
    return eval

def cross_validate(x_list, y_list, w):
    best_model = 0
    max_f1 = 0
    for i in range(4):
        x = x_list[i % 4] + x_list[(i + 1) % 4] + x_list[(i + 2) % 4]
        y = y_list[i % 4] + y_list[(i + 1) % 4] + y_list[(i + 2) % 4]
        x_resampled, y_resampled = SMOTE().fit_sample(x, y)
        for j in range (5):
            w, gradient = batch_train(w, x_resampled, y_resampled)
            eval = test_accuracy(w, x_list[(i + 3) % 4], y_list[(i + 3) % 4])
            if max_f1 < eval.F1():
                best_model = w
        eval = test_accuracy(best_model, x_list[(i + 3) % 4], y_list[(i + 3) % 4])
        print("The training accuracy is: %f" % eval.accuracy())
        print("The training precision is: %f" % eval.precision())
        print("The training recall is: %f" % eval.recall())
        print("The training F1 measure is: %f" % eval.F1())
    return best_model

def individual_validate(x_list, y_list, w):
    best_model = 0
    max_f1 = 0
    x_resampled, y_resampled = SMOTE().fit_sample(x_list, y_list)
    for j in range (50):
        w, gradient = batch_train(w, x_resampled, y_resampled)
        eval = test_accuracy(w, x_list, y_list)
        if max_f1 < eval.F1():
            best_model = w
    eval = test_accuracy(best_model, x_list, x_list)
    print("The training accuracy is: %f" % eval.accuracy())
    print("The training precision is: %f" % eval.precision())
    print("The training recall is: %f" % eval.recall())
    print("The training F1 measure is: %f" % eval.F1())
    return best_model

def classify(w, testing_data, file="test.csv"):
    f = open(file, "w")
    pos_count = 0
    for row in testing_data:
        pred, prob = predict_prob(w, np.matrix(row).T)
        if prob >= 0.25:
            pos_count += 1
        f.write("%f,%d\n" % (prob, pred))

    f.close()
    print "Classification file saved as '" + file + "'"
    print "Found %d positives out of %d data points!" % (pos_count, len(testing_data))

if __name__ == "__main__":
    count = 0
    training_acc = []
    testing_acc = []   

    x_list = []
    y_list = []

    # Subject A = 1, B = 4, C = 6, D = 9
    window_list, window_label = dp.get_window_data("train_data/Subject_1.csv",
                                                   "train_data/list_1.csv")
    x_list.append(window_list)
    y_list.append(window_label)

    window_list, window_label = dp.get_window_data("train_data/Subject_4.csv",
                                                   "train_data/list_4.csv")
    x_list.append(window_list)
    y_list.append(window_label)

    window_list, window_label = dp.get_window_data("train_data/Subject_6.csv",
                                                   "train_data/list_6.csv")
    x_list.append(window_list)
    y_list.append(window_label)

    window_list, window_label = dp.get_window_data("train_data/Subject_9.csv",
                                                   "train_data/list_9.csv")
    x_list.append(window_list)
    y_list.append(window_label)

    w = np.matrix([0]*49).T
    w = cross_validate(x_list, y_list, w)
    test_data = dp.get_test_data("test_data/general_test_instances.csv")
    classify(w, test_data, file="general_pred1.csv")

    # Individual 1
    window_list, window_label = dp.get_window_data("train_data/Subject_2_part1.csv",
                                                   "train_data/list2_part1.csv")
    w = individual_validate(window_list, window_label, w)
    test_data = dp.get_test_data("test_data/subject2_instances.csv")
    classify(w, test_data, file="individual1_pred1.csv")

    # Individual 2
    window_list, window_label = dp.get_window_data("train_data/Subject_7_part1.csv",
                                                   "train_data/list_7_part1.csv")
    w = individual_validate(window_list, window_label, w)
    test_data = dp.get_test_data("test_data/subject7_instances.csv")
    classify(w, test_data, file="individual2_pred1.csv")


