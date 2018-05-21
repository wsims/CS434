import numpy as np
import random
import math

class Cluster(object):
    def __init__(self, mean):
        self.mean = mean
        self.cluster = []

    def add_data(self, obs):
        self.cluster.append(obs)

    def distance(self, obs):
        distance = 0
        for i in range(len(obs)):
            distance += (obs[i] - self.mean[i])**2
    
        return distance

    def get_SSE(self):
        SSE = 0
        for value in self.cluster:
            SSE += self.distance(value)
        return SSE

    def get_new_mean(self):
        new_mean = []
        for i in range(len(self.cluster[0])):
            sum = 0
            for value in self.cluster:
                sum += value[i]
            sum /= float(len(self.cluster))
            new_mean.append(sum)
        return new_mean

class KCluster(object):
    def __init__(self, means):
        self.clusters = [Cluster(means[i]) for i in range(len(means))]

    def _get_closest_cluster(self, obs):
        min_distance = float("inf")
        min_index = None
        for i, cluster in enumerate(self.clusters):
            distance = cluster.distance(obs)
            if distance < min_distance:
                min_distance = distance
                min_index = i
    
        return min_index
    
    def cluster_data(self, data):
        for row in data:
            index = self._get_closest_cluster(row)
            self.clusters[index].add_data(row)

    def get_SSE(self):
        SSE = 0
        for clust in self.clusters:
            SSE += clust.get_SSE()
        return SSE

    def get_new_means(self):
        means = []
        for value in self.clusters:
            mean = value.get_new_mean()
            means.append(mean)
        return means

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

def k_means(data, k):
    # Pick seeds
    means = get_k_seeds(data, k)

    # Create initial cluster
    cluster_set = KCluster(means)
    cluster_set.cluster_data(data)
    new_SSE = cluster_set.get_SSE()
    old_SSE = -1

    # Create new clusters until SSE converges
    while old_SSE != new_SSE:
        means = cluster_set.get_new_means()
        cluster_set = KCluster(means)
        cluster_set.cluster_data(data)
        old_SSE = new_SSE
        new_SSE = cluster_set.get_SSE()
    return new_SSE

if __name__ == '__main__':
    data = get_data()
    SSE = k_means(data, 10)
    print SSE
    # print data.shape

