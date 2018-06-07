import numpy as np

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from assignment_2 import decision_tree as dt
import data_process as dp

def dtree_data_trans(data_list, label_list):
    full_array = []

    for i, row in enumerate(data_list):
        label_value = 1
        if label_list[i] == 0:
            label_value = -1
        full_array.append([label_value] + row)

    return full_array

if __name__ == "__main__":
    window_list, window_label = dp.get_window_data("train_data/Subject_1.csv",
                                                   "train_data/list_1.csv")

    data = dtree_data_trans(window_list, window_label)

    d = 10
    for d in range(10, 20):
        dtree = dt.build_tree(data, d)

        train_accuracy = dt.compute_accuracy(dtree, data)
        print("For a tree of depth %d:" % d)
        print("The training accuracy is: %f" % train_accuracy)

