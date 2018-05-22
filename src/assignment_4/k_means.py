import numpy as np
import random
import math
import matplotlib.pyplot as plt

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
        if len(self.cluster) == 0:
            new_mean = self.mean
        else:
            for i in range(len(self.mean)):
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
    new_means = get_k_seeds(data, k)

    # Create initial cluster
    cluster_set = KCluster(new_means)
    cluster_set.cluster_data(data)
    SSE_list = [cluster_set.get_SSE()]
    old_means = new_means
    new_means = cluster_set.get_new_means()

    # Create new clusters until means converge
    while cmp(old_means, new_means) != 0:
        cluster_set = KCluster(new_means)
        cluster_set.cluster_data(data)
        SSE_list.append(cluster_set.get_SSE())
        old_means = new_means
        new_means = cluster_set.get_new_means()
    return SSE_list

if __name__ == '__main__':
    # Data Retrieval
    data = get_data()

    # Question 2.1
    SSE_list = k_means(data, 2)
    print("SSE convergence list for k=2:")
    print(SSE_list)
    cluster_updates = range(len(SSE_list))

    # Question 2.1 plot
    plt.figure(1)
    plt.plot(cluster_updates, SSE_list, '-b')
    plt.xlabel('Number of cluster updates')
    plt.ylabel('SSE')
    plt.title('K-means algorithm SSE convergence with k=2')
    plt.savefig("k2_means.png")
    print("Question 2.1 plot saved as 'k2_means.png'\n")

    # Question 2.2
    SSE_list = []
    for i in range(2, 11):
        print("Running k-means for k = %d:" % i)
        SSE_min = float('inf')
        for j in range(10):
            print("Performing trial %d..." % (j+1))
            SSE_converge_list = k_means(data, i)
            SSE = SSE_converge_list[len(SSE_converge_list)-1]
            if SSE < SSE_min:
                SSE_min = SSE
        SSE_list.append(SSE_min)
    k_values = range(2, 11)
    print("List of SSE for k-values 2-10:")
    print(SSE_list)

    # Question 2.2 plot
    plt.figure(2)
    plt.plot(k_values, SSE_list, '-b')
    plt.xlabel('Number of clusters')
    plt.ylabel('Lowest SSE out of 10 trials')
    plt.title('K-means algorithm SSE vs. number of clusters k')
    plt.savefig("k_clusters_2_2.png")
    print("Question 2.2 plot saved as 'k_clusters_2_2.png'")

