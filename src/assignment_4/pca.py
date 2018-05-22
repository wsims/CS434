import numpy as np

def get_data(file='data-1.txt'):
    f = open(file, 'r')
    data_list = []

    for line in f:
        value_list = map(float, line.split(','))
        data_list.append(value_list)

    data_mat = np.matrix(data_list)
    return data_mat

def get_mean(data):
    mean = np.sum(data, axis=0)
    mean = mean / len(data)
    return np.matrix(mean).T

def get_covariance(data, mean):
    n, d = data.shape
    cov_mat = np.zeros((d, d))
    for row in data:
        x = row.T
        diff = x - mean
        cov_mat = cov_mat + diff * diff.T

    cov_mat = cov_mat / len(data)
    return cov_mat

def get_eigen(cov_mat):
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    eig_list = []
    for i, eig_val in enumerate(eigenvalues):
        eig_list.append([eig_val, eigenvectors[:, i]])
    eig_list.sort(key=lambda x: x[0])
    eig_list.reverse()
    return eig_list

if __name__ == '__main__':
    data = get_data()
    mean = get_mean(data)
    cov_mat = get_covariance(data, mean)
    eig_list = get_eigen(cov_mat)
    for i in range(10):
        #print(eig_list[i][0])
        print("Eigenvalue %d: %f" % (i+1, eig_list[i][0].real))
