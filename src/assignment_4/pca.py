"""
Principal Component Analysis Implementation

Uses Principal Component Analysis to perform dimension
reduction on a set of 28x28 images to reduce the dimensionality
from 784 features down to 10.  The mean image of the data set
is stored as 'mean.png'.  The 10 eigenvectors produced are
stored as 'eigenvector{n}.png' for n = 1, 2,..., 10.  The images
that have the largest value in each dimension are stored as
'dimension{n}.png'

This script is used to satisfy part 3 in its entirety for assignment 4.

Usage:
    $ python pca.py

"""

import numpy as np
from PIL import Image

def get_data(file='data-1.txt'):
    """Extract data from a file and return it
    in the form of a numpy matrix.
    """
    f = open(file, 'r')
    data_list = []

    for line in f:
        value_list = map(float, line.split(','))
        data_list.append(value_list)

    data_mat = np.matrix(data_list)
    return data_mat

def get_mean(data):
    """Returns the mean of the rows in the data set.

    Inputs:
        data (2D numpy matrix): each row is a data point

    Outputs:
        mean (numpy column vector): mean of all data points

    """
    mean = np.sum(data, axis=0)
    mean = mean / len(data)
    return np.matrix(mean).T

def get_covariance(data, mean):
    """Returns the covariance matrix for the data set.

    Inputs:
        data (2D numpy matrix): each row is a data point
        mean (numpy column vector): mean of all data points

    Outputs:
        cov_mat (2D numpy matrix): Covariance matrix for the data set

    """
    n, d = data.shape
    cov_mat = np.zeros((d, d))
    for row in data:
        x = row.T
        diff = x - mean
        cov_mat = cov_mat + diff * diff.T

    cov_mat = cov_mat / len(data)
    return cov_mat

def get_eigen(cov_mat):
    """Returns a sorted list of associated eigenvalues and eigenvectors
    for a given covariance matrix.

    Inputs:
        cov_mat (2D numpy matrix): Covariance matrix for the data set

    Outputs:
        eig_list (list): Sorted list of eigen values and eigen vectors.
            Eigenvalues occupy the first entry in each sublist, while
            Eigenvectors occupy the second.  Eigenvectors are in the form
            of a numpy column vector.

    """
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    eig_list = []
    for i, eig_val in enumerate(eigenvalues):
        eig_list.append([eig_val, eigenvectors[:, i]])
    eig_list.sort(key=lambda x: x[0])
    eig_list.reverse()
    return eig_list

def save_image(vector, file):
    """Translates a 784x1 numpy matrix into a scaled image and saves it.

    Inputs:
        vector (numpy column vector): The data we wish to save as an image.
        file (str): File name we wish to save the image as.

    """
    vector_temp = vector.real # get only the real components from each entry
    max = vector_temp.max()
    vector_temp = vector_temp * (255.0 / max)
    im = Image.fromarray(vector_temp.reshape(28, 28))
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(file)
    print "Image saved as '" + file + "'"

def get_projection_mag(image, eigenvector, mean):
    """Returns the magnitude of the projection of an image onto
    an eigenvector.

    Inputs:
        image (numpy column vector): The data point we wish to project
            onto the eigenvector.
        eigenvector (numpy column vector): The eigenvector we will project
            onto.
        mean (numpy column vector): The mean of the data set.  This is used
            to translate and normalize image vectors.

    Outputs:
        magnitude (float): The magnitude of the projection of the image
            onto the eigenvector.

    """
    translation = image - mean
    magnitude = np.inner(translation.T, eigenvector.T).item(0)
    return magnitude

if __name__ == '__main__':
    data = get_data()
    mean = get_mean(data)
    cov_mat = get_covariance(data, mean)
    eig_list = get_eigen(cov_mat)

    # Question 3.1
    for i in range(10):
        print("Eigenvalue %d: %f" % (i+1, eig_list[i][0].real))

    # Question 3.2
    save_image(mean, "mean.png")

    for i in range(10):
        save_image(eig_list[i][1], "eigenvector" + str(i+1) + ".png")

    # Question 3.3
    image_list = [[-float('inf'), None] for i in range(10)]
    for row in data:
        for i in range(10):
            projection_mag = get_projection_mag(row.T, eig_list[i][1].real, mean)
            if projection_mag > image_list[i][0]:
                image_list[i][0] = projection_mag
                image_list[i][1] = row

    for i, image in enumerate(image_list):
        save_image(image[1].T, "dimension" + str(i+1) + ".png")


