import numpy as np
from PIL import Image

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

def save_image(vector, file):
    vector_temp = vector.real # get only the real components from each entry
    max = vector_temp.max()
    vector_temp = vector_temp * (255.0 / max)
    im = Image.fromarray(vector_temp.reshape(28, 28))
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(file)
    print "Image saved as '" + file + "'"

def get_projection_mag(image, eigenvector, mean):
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


