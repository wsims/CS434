"""Data Processing Module

Contains functionality to preprocess the data for use by
the classifiers.

"""
import numpy as np

def get_data(data_file):
    """Read the data file and extract the raw feature data and class labels.

    Inputs:
        data_file (str): Name of the csv file to read.

    Outputs:
        data_list (2D Array): Each row contains an observation, and each
            column contains data for a specific feature.
        label_list (Python list): List of class labels (0 or 1).

    """
    df = open(data_file, 'r')
    data_list = []
    label_list = []

    for line in df:
        value_list = line.split(',')
        if len(value_list) > 9:
            label_list.append(int(value_list.pop()))  # Remove label
        value_list.pop(0)   # Remove time stamp from data
        value_list.pop()    # Remove night time dummy variable (redundant info)
        data_list.append(map(float, value_list))

    df.close()

    return data_list, label_list

def get_test_data(data_file):
    """Reads and reformats test data for use."""
    df = open(data_file, 'r')
    data_list = []

    for line in df:
        formatted_row = []
        value_list = line.split(',')
        for i in range(7):
            for j in range(7):
                formatted_row.append(value_list[7+i+7*j])
        data_list.append(map(float, formatted_row))

    df.close()

    return data_list

def get_indices(index_file):
    """Reads the index file and return it as a Python List"""
    index_f = open(index_file, 'r')
    index_list = []

    for line in index_f:
        index_list.append(int(line))

    index_f.close()

    return index_list

def get_window_data(data_file, index_file):
    """Reads the data and reformats it into 30 minute windows
    for each observation.

    Inputs:
        data_file (str): Name of the file containing the feature data.
        index_file (str): Name of the file containing the data indices.

    Outputs:
        window_list (2D Array): Array containing the 30 minute window data.
        window_label (Python list): 1D array containing the class labels for
            each window of data.

    """
    raw_data, labels = get_data(data_file)
    indices = get_indices(index_file)
    sequence = 1

    window_list = []
    window_label = []

    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            sequence += 1
        else:
            sequence = 1

        if sequence >= 7:
            window = []
            for j in range(7):
                window += raw_data[i - 6 + j]
            window_list.append(window)
            window_label.append(labels[i])

    return window_list, window_label

if __name__ == "__main__":
    w_list, w_label = get_window_data("train_data/Subject_1.csv", 
                                      "train_data/list_1.csv")

    test_list = get_test_data("test_data/general_test_instances.csv")

    print len(w_label)
    print w_list[0]

    print test_list[0]

    print len(w_list[0])
    print len(test_list[0])
