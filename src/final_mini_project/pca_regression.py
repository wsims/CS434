import numpy as np

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import data_process as dp
import performance as perf
from assignment_4 import pca

CROSS_VALIDATE = 0
CLASSIFY_TEST_DATA = 1

#MODE = CROSS_VALIDATE
MODE = CLASSIFY_TEST_DATA

def get_data_matrix(w_list):
    new_list = []
    for row in w_list:
        new_list.append([1.0] + row)
    return np.matrix(new_list)

def get_training_data(w_list, w_label):
    sampled_W_list = []
    sampled_W_label = []
    sampled_W_list += w_list
    sampled_W_label += w_label
    pos = 0
    total = 0
    for i in range(len(w_label)):
        if w_label[i] == 1:
            pos += 1
        total += 1
    oversample = int((0.163*total - pos)/((1-0.163)*pos))
    if oversample > 0:
        for i in range(len(w_label)):
            if w_label[i] == 1:
                for j in range(oversample):
                    sampled_W_list.append(w_list[i])
                    sampled_W_label.append(w_label[i])

    X = get_data_matrix(sampled_W_list)
    Y = np.matrix(sampled_W_label).T

    return X, Y


def get_principal_components(X):
    mean = pca.get_mean(X)
    cov_mat = pca.get_covariance(X, mean)
    eig_list = pca.get_eigen(cov_mat)
    return eig_list

def pca_projection(X, eig_list, mean, features):
    new_data_list = []
    for row in X:
        new_row = []
        for i in range(features):
            proj = pca.get_projection_mag(row.T, eig_list[i][1].real, mean)
            new_row.append(proj)
        new_data_list.append(new_row)
    return np.matrix(new_data_list)

def dimension_reduction(train_data, test_data, features=10):
    mean = pca.get_mean(train_data)

    eig_list = get_principal_components(train_data)

    #for i in range(len(eig_list)):
    #    print("Eigenvalue %d: %f" % (i+1, eig_list[i][0]))

    X_pca = pca_projection(train_data, eig_list, mean, features)
    X_pca_test = pca_projection(test_data, eig_list, mean, features)
    return X_pca, X_pca_test



def regress(X, Y):
    return np.linalg.inv(X.T*X)*X.T*Y

def prediction(X_pca_row, W):
    #pca_obs = []
    #for i in range(features):
    #    proj = pca.get_projection_mag(X_i, eig_list[i][1].real, mean)
    #    pca_obs.append(proj)
    #pca_obs = np.matrix(pca_obs).T
    return np.inner(X_pca_row.T, W.T).item(0)

def float_to_binary(pred):
    value = 0
    if pred >= 0.25:
        value = 1
    return value

def cross_validate(data_list, label_list, features):
    eval = None
    for i in range(4):
        print "Validating set %d..." % i
        data = data_list[i%4] + data_list[(i+1)%4] + data_list[(i+2)%4]
        label = label_list[i%4] + label_list[(i+1)%4] + label_list[(i+2)%4]
        data, label = get_training_data(data, label)
        test_data = get_data_matrix(data_list[(i+3)%4])
        test_label = np.matrix(label_list[(i+3)%4]).T

        X_pca, X_pca_test = dimension_reduction(data, test_data, features)

        W = regress(X_pca, label)

        if i == 0:
            eval = compute_accuracy(X_pca_test, test_label, W)
        else:
            eval = eval + compute_accuracy(X_pca_test, test_label, W)

    return eval

def compute_accuracy(data, Y, W):
    eval = perf.PerformanceEval()
    for i, row in enumerate(data):
        eval.add_result(float_to_binary(prediction(row.T, W)),
                        int(Y.item(i)))
    return eval

def classify(training_data, training_label, testing_data, features=10, file="test.csv"):
    train_data, train_label = get_training_data(training_data, training_label)
    test_data = get_data_matrix(testing_data)

    train_pca, test_pca = dimension_reduction(train_data, test_data, features)

    W = regress(train_pca, train_label)

    f = open(file, "w")

    pos_count = 0
    for row in test_pca:
        pred = prediction(row.T, W)
        if pred >= 0.25:
            pos_count += 1
        f.write("%f,%d\n" % (pred, float_to_binary(pred)))

    f.close()
    print "Classification file saved as '" + file + "'"
    print "Found %d positives out of %d data points!" % (pos_count, len(test_pca))


if __name__ == "__main__":

    if MODE == CROSS_VALIDATE:
        data_list, label_list = [], []

        w_list, w_label = dp.get_window_data("train_data/Subject_1.csv",
                                             "train_data/list_1.csv")
        data_list.append(w_list)
        label_list.append(w_label)

        w_list, w_label = dp.get_window_data("train_data/Subject_4.csv",
                                             "train_data/list_4.csv")
        data_list.append(w_list)
        label_list.append(w_label)

        w_list, w_label = dp.get_window_data("train_data/Subject_6.csv",
                                             "train_data/list_6.csv")
        data_list.append(w_list)
        label_list.append(w_label)

        w_list, w_label = dp.get_window_data("train_data/Subject_9.csv",
                                             "train_data/list_9.csv")
        data_list.append(w_list)
        label_list.append(w_label)

        f1_list = []
        for i in range(1, 31):
            print "Running test for %d features" % i
            f1_list.append(cross_validate(data_list, label_list, i).F1())
            print ""

        best_index = f1_list.index(max(f1_list))

        print "Highest F1 score found when using %d features" % (best_index + 1)
        print "F1 score at peak: %f" % f1_list[best_index]

        print f1_list

    elif MODE == CLASSIFY_TEST_DATA:
        train_data, train_label = [], []

        # General Population

        w_list, w_label = dp.get_window_data("train_data/Subject_1.csv",
                                             "train_data/list_1.csv")
        train_data += w_list
        train_label += w_label

        w_list, w_label = dp.get_window_data("train_data/Subject_4.csv",
                                             "train_data/list_4.csv")
        train_data += w_list
        train_label += w_label

        w_list, w_label = dp.get_window_data("train_data/Subject_6.csv",
                                             "train_data/list_6.csv")
        train_data += w_list
        train_label += w_label

        w_list, w_label = dp.get_window_data("train_data/Subject_9.csv",
                                             "train_data/list_9.csv")
        train_data += w_list
        train_label += w_label

        test_data = dp.get_test_data("test_data/general_test_instances.csv")

        classify(train_data, train_label, test_data, file="preds/general_pred3.csv")

        # Individual 1

        train_data, train_label = dp.get_window_data("train_data/Subject_2_part1.csv",
                                                     "train_data/list2_part1.csv")

        test_data = dp.get_test_data("test_data/subject2_instances.csv")

        classify(train_data, train_label, test_data, file="preds/individual1_pred3.csv")

        # Individual 2

        train_data, train_label = dp.get_window_data("train_data/Subject_7_part1.csv",
                                                     "train_data/list_7_part1.csv")

        test_data = dp.get_test_data("test_data/subject7_instances.csv")

        classify(train_data, train_label, test_data, file="preds/individual2_pred3.csv")



    #X_test = get_data_matrix(w_list)
    #Y_test = np.matrix(w_label).T

    #X, Y = get_training_data(w_list, w_label)

    #X_pca, X_pca_test = dimension_reduction(X, X_test)

    #W = regress(X_pca, Y)

    #eval = compute_accuracy(X_pca_test, Y_test, W)
    #print eval.accuracy()
    #print eval.F1()
