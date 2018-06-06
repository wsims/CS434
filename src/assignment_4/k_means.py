"""
K-means Algorithm Implementation

Performs K-means clustering on a set of 28x28 images of handwritten
digits.  For k=2, this script plots the SSE versus the number of
clustering iterations and stores the plot as 'k_means.png'.  This 
script also plots the lowest convergence SSE out of 10 trials for
k = 2, 3, ..., 10 and stores the plot as 'k_clusters_2_2.png'.

This script is used to satisfy part 2 in its entirety for assignment 4.

Usage:
    $ python k_means.py

"""

import numpy as np
import random
import math
#import matplotlib.pyplot as plt

class Cluster(object):
    """Individual Cluster object"""

    def __init__(self, mean):
        self.mean = mean
        self.cluster = []

    def add_data(self, obs):
        """Add a data point to the cluster"""
        self.cluster.append(obs)

    def distance(self, obs, mean=None):
        """Calculate the distance from an observation to the mean value
        of a cluster.

        Inputs:
            obs (numpy column vector): A single data point
            mean (numpy column vector): optional value; uses the 
                cluster mean value if not provided.

        Outputs:
            distance (float): The euclidean distance squared from the
                observation to the mean.
        """
        if mean == None:
            mean = self.mean
        x = obs - mean
        distance = np.dot(x.T, x).item(0)

        return distance

    def get_SSE(self):
        """Returns the SSE for a cluster using the mean of the points
        currently in the cluster.

        """
        SSE = 0
        mean = self.get_new_mean()
        for value in self.cluster:
            SSE += self.distance(value, mean)
        return SSE

    def get_new_mean(self):
        """Calculates the mean of all observations in the cluster"""
        new_mean = []
        if len(self.cluster) == 0:
            new_mean = self.mean # Necessary to deal with emtpy clusters
        else:
            for i in range(len(self.mean)): # iterate over each feature
                sum = 0
                for value in self.cluster: # iterate over each data point
                    sum += value[i]
                sum /= float(len(self.cluster))
                new_mean.append(sum)
        return new_mean

class KCluster(object):
    """Full KCluster object"""

    def __init__(self, means):
        self.clusters = [Cluster(means[i]) for i in range(len(means))]

    def _get_closest_cluster(self, obs):
        """Returns the index of the closest cluster to an observation"""
        min_distance = float("inf")
        min_index = None
        for i, cluster in enumerate(self.clusters):
            distance = cluster.distance(obs)
            if distance < min_distance:
                min_distance = distance
                min_index = i

        return min_index

    def cluster_data(self, data):
        """Sort data set into clusters.

        Inputs:
            data (2D numpy matrix): Each row is a data point.

        """
        for row in data:
            index = self._get_closest_cluster(row)
            self.clusters[index].add_data(row)

    def get_SSE(self):
        """Calculate the sum of the SSE for all clusters"""
        SSE = 0
        for clust in self.clusters:
            SSE += clust.get_SSE()
        return SSE

    def get_new_means(self):
        """Returns a list of new means for each cluster"""
        means = []
        for value in self.clusters:
            mean = value.get_new_mean()
            means.append(mean)
        return means

def get_data(file='data-1.txt'):
    """Extract data from a file and return it
    in the form of a numpy matrix.
    """
    f = open(file, 'r')
    data_list = []

    for line in f:
        value_list = map(int, line.split(','))
        data_list.append(np.array(value_list).T)

    return data_list

def get_k_seeds(data, k):
    """Randomly selects k data points from a data set to use as initial
    means.

    Inputs:
        data (2D numpy matrix): each row is a data point
        k (int): number of points to select

    Outputs:
        means (list of numpy vectors): List of randomly selected intial means.

    """
    means = []
    length = len(data)
    for i in range(k):
        seed_index = random.randint(0, length-1)
        means.append(data[seed_index])
    return means

def k_means(data, k):
    """Perform k_means clustering on a data set"""
    # Pick seeds
    means = get_k_seeds(data, k)

    # Create initial cluster
    cluster_set = KCluster(means)
    cluster_set.cluster_data(data)
    SSE_list = [-1, cluster_set.get_SSE()]

    # Create new clusters until means converge
    while abs(SSE_list[-2] - SSE_list[-1]) > 1E-10:
        means = cluster_set.get_new_means()
        cluster_set = KCluster(means)
        cluster_set.cluster_data(data)
        SSE_list.append(cluster_set.get_SSE())

    SSE_list.pop(0)
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
    #plt.figure(1)
    #plt.plot(cluster_updates, SSE_list, '-b')
    #plt.xlabel('Number of clustering iterations')
    #plt.ylabel('SSE')
    #plt.title('K-means algorithm SSE convergence with k=2')
    #plt.savefig("k2_means.png")
    #print("Question 2.1 plot saved as 'k2_means.png'\n")

    # Question 2.2
    SSE_list = []
    for i in range(2, 11):
        print("Running k-means for k = %d:" % i)
        SSE_min = float('inf')
        for j in range(10):
            print("Performing trial %d..." % (j+1))
            SSE_converge_list = k_means(data, i)
            SSE = SSE_converge_list[-1]
            if SSE < SSE_min:
                SSE_min = SSE
        SSE_list.append(SSE_min)
    k_values = range(2, 11)
    print("List of SSE for k-values 2-10:")
    print(SSE_list)

    # Question 2.2 plot
    #plt.figure(2)
    #plt.plot(k_values, SSE_list, '-b')
    #plt.xlabel('Number of clusters')
    #plt.ylabel('Lowest SSE out of 10 trials')
    #plt.title('K-means algorithm SSE vs. number of clusters k')
    #plt.savefig("k_clusters_2_2.png")
    #print("Question 2.2 plot saved as 'k_clusters_2_2.png'")

