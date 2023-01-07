import numpy as np
import copy

from Distance import Distance


class KMeans:
    def __init__(self, dataset, K=2):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_centers stores the cluster mean vectors for each cluster in a dictionary
        self.cluster_centers = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class

    @staticmethod
    def compareDict(dict1, dict2):
        """Compare two dictionaries"""
        for i in range(len(dict1.keys())):
            for j in range(2):
                if dict1[i][j] != dict2[i][j]:
                    return False
        return True

    def calculateLoss(self):
        """Loss function implementation of Equation 1"""
        # calculate the loss function of each cluster with the cluster center
        loss = 0
        for i in range(len(self.cluster_centers)):
            for j in range(len(self.clusters[i])):
                distance = Distance.calculateMinkowskiDistance(self.clusters[i][j], self.cluster_centers[i], 2)
                loss += distance ** 2
        return loss

    def run(self):
        """Kmeans algorithm implementation"""
        # initialize cluster centers randomly from the dataset points
        initial_centers = np.random.choice(self.dataset.shape[0], self.K, replace=False)
        self.cluster_centers = {i: self.dataset[initial_centers[i]] for i in range(self.K)}

        # create empty dictionary to store the previous cluster centers
        previous_centers = {i: [0] for i in range(self.K)}

        # run the Kmeans algorithm until convergence
        dict1 = None
        while self.compareDict(self.cluster_centers, previous_centers) is False:

            previous_centers = copy.deepcopy(self.cluster_centers)
            # update the clusters
            for i in range(self.dataset.shape[0]):
                min_dist = float('inf')  # initialize the minimum distance to infinity
                min_cluster = -1  # initialize the cluster index to -1
                for j in range(self.K):
                    dist = Distance.calculateMinkowskiDistance(self.dataset[i], self.cluster_centers[j], 2)
                    if dist < min_dist:  # update the minimum distance and the cluster index
                        min_dist = dist
                        min_cluster = j
                self.clusters[min_cluster].append(self.dataset[i])  # add the data point to the cluster

            dict1 = copy.deepcopy(self.clusters)
            # update the cluster centers
            for i in range(self.K):  # for each cluster
                if len(self.clusters[i]) != 0:
                    self.cluster_centers[i] = np.mean(self.clusters[i], axis=0)  # update the cluster center
                self.clusters[i] = []  # clear the cluster

        self.clusters = dict1
        return self.cluster_centers, self.clusters, self.calculateLoss()
