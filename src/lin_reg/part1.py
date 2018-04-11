import numpy as np

def get_housing_data(file):
    y_list = []
    x_list = []
    f = open(file, 'r')

    for line in f:
        value_list = map(float, line.split())
        y_list.append(value_list[13])
        value_list.insert(0, 1)
        x_list.append(value_list[:14])

    f.close()
    y = np.matrix(y_list)
    x = np.matrix(x_list)
    y = y.T
    return x, y

def get_data_no_dummy(file):
    y_list = []
    x_list = []
    f = open(file, 'r')

    for line in f:
        value_list = map(float, line.split())
        y_list.append(value_list[13])
        x_list.append(value_list[:13])

    f.close()
    y = np.matrix(y_list)
    x = np.matrix(x_list)
    y = y.T
    return x, y

def get_ASE(file, w):
    x, y = get_housing_data(file)
    count = 0
    f = open(file, 'r')
    for line in f:
        count += 1

    SSE = (y-x*w).T*(y-x*w)
    SSE = SSE.item(0)
    ASE = SSE/count

    return ASE

def get_ASE_no_dummy(file, w):
    x, y = get_data_no_dummy(file)
    count = 0
    f = open(file, 'r')
    for line in f:
        count += 1

    SSE = (y-x*w).T*(y-x*w)
    SSE = SSE.item(0)
    ASE = SSE/count

    return ASE

def main():
    # Subsection 1
    print "Vector w from subsection 1:"
    x, y = get_housing_data('housing_train.txt')
    w = np.linalg.inv(x.T*x)*x.T*y
    print w
    print '\n'

    # Subsection 2
    print "Results from subsection 2:"
    ASE_train = get_ASE('housing_train.txt', w)
    ASE_test = get_ASE('housing_test.txt', w)
    print("Training data ASE: %f" % ASE_train)
    print("Testing data ASE: %f" % ASE_test)
    print '\n'

    # Subsection 3
    print "Results from subsection 3:"
    x, y = get_data_no_dummy('housing_train.txt')
    w = np.linalg.inv(x.T*x)*x.T*y
    print w
    ASE_train = get_ASE_no_dummy('housing_train.txt', w)
    ASE_test = get_ASE_no_dummy('housing_test.txt', w)
    print("Training data ASE: %f" % ASE_train)
    print("Testing data ASE: %f" % ASE_test)

main()
