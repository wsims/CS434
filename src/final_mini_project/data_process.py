import numpy as np

def get_data(data_file):
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

def get_indices(index_file):
    index_f = open(index_file, 'r')
    index_list = []

    for line in index_f:
        index_list.append(int(line))

    index_f.close()

    return index_list

def get_window_data(data_file, index_file):
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

    return np.matrix(window_list), np.matrix(window_label).T

if __name__ == "__main__":
    w_list, w_label = get_window_data("train_data/Subject_1.csv", 
                                      "train_data/list_1.csv")

    print w_label


