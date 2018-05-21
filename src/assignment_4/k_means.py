import numpy as np
import random
import math

def get_data(file='unsupervised.txt'):
    f = open(file, 'r')
    data_list = []

    for line in f:
        value_list = map(int, line.split(','))
        data_list.append(value_list)

    # data_mat = np.matrix(data_list)
    return data_list

def get_k_seeds(data, k):
    means = []
    length = len(data)
    for i in range(k):
        seed_index = random.randint(0, length-1)
        means.append(data[seed_index])
    return means

def distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i])**2

    return distance


def get_closest_cluster(row, means):
    min_distance = float("inf")
    min_index = None
    for i, mew in enumerate(means):
        distance = distance(row, mew)
        if distance < min_distance:
            min_distance = distance
            min_index = i

    return min_index

def get_clusters(data, means):
    clusters = [[] for i in range(len(means))]
    for row in data:
        index = get_closest_cluster(row, means)
        clusters[index].append(row)
    return clusters

def get_SSE(cluster_list, mean_list):
    SSE = 0
    for i, cluster in enumerate(cluster_list):
        for row in cluster:
            SSE += distance(row, mean_list[i])
    return SSE

def k_means(data, k):
    # Pick seeds
    means = get_k_seeds(data, k)
    clusters = get_clusters(data, means)

if __name__ == '__main__':
    data = get_data()
    # print data.shape

