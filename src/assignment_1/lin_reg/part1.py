"""
Usage: python part1.py

Performs a linear regression to produce an optimal weight vector
which can be used to predict the median value of housing in an
area (in thousands) based on 13 different features.  Determines
the weight vector when a dummy variable is included and excluded,
computes the ASE for both the training data and testing data when
using the weight vector determined with a dummy variable, and plots
the effect of adding additional random features on the ASEs for the
training data and testing data.

This script is used to satisfy part 1 in its entirety for assignment 1.

"""
import numpy as np
#import matplotlib.pyplot as plt

def get_housing_data(file, d=0):
    """Reads feature and output data from a file and returns
    a feature matrix (with dummy variable included) and an output
    matrix.  Can also add d additional random features to the
    feature matrix.

    Inputs:
        file -- the name of a text file containing space delimited housing
            data.
        d -- number of additional, random, normally distributed features to
            add to the data set.

    Outputs:
        x -- a numpy matrix object containing all feature data for the set.
            Each row represents a single observation and each column represents
            a single feature.
        y -- a numpy matrix object in the form of a column vector.
            This vector contains the true median housing value for each
            area.

    """
    y_list = []
    x_list = []
    lines = 0
    count = 0
    f = open(file, 'r')

    for line in f:
        value_list = map(float, line.split())
        y_list.append(value_list[13])
        if d > 0:
            for i in range(0, d):
                value_list.insert(0, np.random.randn())
        value_list.insert(0, 1)
        x_list.append(value_list[:len(value_list)-1])

    f.close()
    y = np.matrix(y_list)
    x = np.matrix(x_list)
    y = y.T
    return x, y

def get_data_no_dummy(file):
    """Reads feature and output data from a file and returns
    a feature matrix (without a dummy variable included) and an output
    matrix.

    Inputs:
        file -- the name of a text file containing space delimited housing
            data.

    Outputs:
        x -- a numpy matrix object containing all feature data for the set.
            Each row represents a single observation and each column represents
            a single feature.
        y -- a numpy matrix object in the form of a column vector.
            This vector contains the true median housing value for each
            area.

    """
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
    """Uses a learned weight vector to predict the
    mean housing value in an area, then compares it
    to the actual mean housing value in order to calculate
    the ASE.

    Inputs:
        file -- the name of a text file containing space delimited housing
            data.
        w -- a numpy matrix object in the form of a column vector.  This
            is the weight vector determined by using the training data.

    Outputs:
        ASE -- the average squared error for the data set contained in the
            text file.

    """
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
    """Uses a learned weight vector to predict the
    mean housing value in an area, then compares it
    to the actual mean housing value in order to calculate
    the ASE. This learned weight vector was derived without
    the inclusion of a dummy variable.

    Inputs:
        file -- the name of a text file containing space delimited housing
            data.
        w -- a numpy matrix object in the form of a column vector.  This
            is the weight vector determined by using the training data.

    Outputs:
        ASE -- the average squared error for the data set contained in the
            text file.

    """
    x, y = get_data_no_dummy(file)
    count = 0
    f = open(file, 'r')
    for line in f:
        count += 1

    SSE = (y-x*w).T*(y-x*w)
    SSE = SSE.item(0)
    ASE = SSE/count

    return ASE

def get_ASE_rand(file, w, x, y):
    """Uses a learned weight vector to predict the
    mean housing value in an area, then compares it
    to the actual mean housing value in order to calculate
    the ASE.  This version is used for data sets with additional
    random features included.

    Inputs:
        file -- the name of a text file containing space delimited housing
            data.
        w -- a numpy matrix object in the form of a column vector.  This
            is the weight vector determined by using the training data.
        x -- a numpy matrix object containing all feature data for the set.
            Each row represents a single observation and each column represents
            a single feature.
        y -- a numpy matrix object in the form of a column vector.
            This vector contains the true median housing value for each
            area.

    Outputs:
        ASE -- the average squared error for the data set contained in the
            text file.

    """
    count = 0
    f = open(file, 'r')
    for line in f:
        count += 1

    SSE = (y-x*w).T*(y-x*w)
    SSE = SSE.item(0)
    ASE = SSE/count

    return ASE

def main():
    """The main routine"""
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
    print '\n'

    # Subsection 4
    print "Results from subsection 4:"
    x, y = get_housing_data('housing_train.txt', 2)
    w = np.linalg.inv(x.T*x)*x.T*y
    print "d=2 Vector weights"
    print w
    x_test, y_test = get_housing_data('housing_test.txt', 2)
    ASE_train = get_ASE_rand('housing_train.txt', w, x, y)
    ASE_test = get_ASE_rand('housing_test.txt', w, x_test, y_test)
    print("Training data ASE: %f" % ASE_train)
    print("Testing data ASE: %f" % ASE_test)
    print '\n'

    # Plot with respect to D
    print 'Plotting random feature data from d=0 to 200.'
    rand_ASE_train = []
    rand_ASE_test = []
    for i in range(0, 200, 2):
        x, y = get_housing_data('housing_train.txt', i)
        w = np.linalg.inv(x.T*x)*x.T*y
        x_test, y_test = get_housing_data('housing_test.txt', i)
        rand_ASE_train.append(get_ASE_rand('housing_train.txt', w, x, y))
        rand_ASE_test.append(get_ASE_rand('housing_test.txt', w, x_test, y_test))

    runs = range(0, 200, 2)
    # plt.plot(runs, rand_ASE_train, '-b', label='training data')
    # plt.plot(runs, rand_ASE_test, '-r', label='testing data')
    # plt.legend(loc='lower right')
    # plt.xlabel('d (Number of random features)')
    # plt.ylabel('Average Standard Error')
    # plt.title('Training and Testing ASEs as a function of d')
    # plt.savefig("part1.png")
    # print 'Plot saved as "part1.png"'

if __name__ == '__main__':
    main()
