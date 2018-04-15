import numpy as np
import matplotlib.pyplot as plt

def get_housing_data(file, d):
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
    x, y = get_housing_data(file, 0)
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

def get_ASE_rand(file, w, x, y):
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
    x, y = get_housing_data('housing_train.txt', 0)
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
    print "D=2 Vector weights"
    print w
    print '\n'
    x_test, y_test = get_housing_data('housing_test.txt', 2)
    ASE_train = get_ASE_rand('housing_train.txt', w, x, y)
    ASE_test = get_ASE_rand('housing_test.txt', w, x_test, y_test)
    print("Training data ASE: %f" % ASE_train)
    print("Testing data ASE: %f" % ASE_test)
    print '\n'


    x, y = get_housing_data('housing_train.txt', 4)
    w = np.linalg.inv(x.T*x)*x.T*y
    print "D=4 Vector weights"
    print w
    print '\n'
    x_test, y_test = get_housing_data('housing_test.txt', 4)
    ASE_train = get_ASE_rand('housing_train.txt', w, x, y)
    ASE_test = get_ASE_rand('housing_test.txt', w, x_test, y_test)
    print("Training data ASE: %f" % ASE_train)
    print("Testing data ASE: %f" % ASE_test)
    print '\n'


    x, y = get_housing_data('housing_train.txt', 6)
    w = np.linalg.inv(x.T*x)*x.T*y
    print "D=6 Vector weights"
    print w
    print '\n'
    x_test, y_test = get_housing_data('housing_test.txt', 6)
    ASE_train = get_ASE_rand('housing_train.txt', w, x, y)
    ASE_test = get_ASE_rand('housing_test.txt', w, x_test, y_test)
    print("Training data ASE: %f" % ASE_train)
    print("Testing data ASE: %f" % ASE_test)
    print '\n'

    x, y = get_housing_data('housing_train.txt', 8)
    w = np.linalg.inv(x.T*x)*x.T*y
    print "D=8 Vector weights"
    print w
    print '\n'
    x_test, y_test = get_housing_data('housing_test.txt', 8)
    ASE_train = get_ASE_rand('housing_train.txt', w, x, y)
    ASE_test = get_ASE_rand('housing_test.txt', w, x_test, y_test)
    print("Training data ASE: %f" % ASE_train)
    print("Testing data ASE: %f" % ASE_test)
    print '\n'


    x, y = get_housing_data('housing_train.txt', 10)
    w = np.linalg.inv(x.T*x)*x.T*y
    print "D=10 Vector weights"
    print w
    print '\n'
    x_test, y_test = get_housing_data('housing_test.txt', 10)
    ASE_train = get_ASE_rand('housing_train.txt', w, x, y)
    ASE_test = get_ASE_rand('housing_test.txt', w, x_test, y_test)
    print("Training data ASE: %f" % ASE_train)
    print("Testing data ASE: %f" % ASE_test)

    # Plot with respect to D
    rand_ASE_train = []
    rand_ASE_test = []
    for i in range(0, 200, 2):
        x, y = get_housing_data('housing_train.txt', i)
        w = np.linalg.inv(x.T*x)*x.T*y
        x_test, y_test = get_housing_data('housing_test.txt', i)
        rand_ASE_train.append(get_ASE_rand('housing_train.txt', w, x, y))
        rand_ASE_test.append(get_ASE_rand('housing_test.txt', w, x_test, y_test))

    runs = range(0, 200, 2)
    plt.plot(runs, rand_ASE_train, '-b', label='training data')
    plt.plot(runs, rand_ASE_test, '-r', label='testing data')
    plt.legend(loc='lower right')
    plt.xlabel('D')
    plt.ylabel('Average Standard Error')
    plt.title('Training and Testing ASEs as a function of D')
    plt.show()

main()
