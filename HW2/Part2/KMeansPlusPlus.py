import copy

import numpy as np
import math
from Distance import Distance

class KMeansPlusPlus:
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
            for j in range(len(dict1[i])):
                if dict1[i][j] != dict2[i][j]:
                    return False
        return True

    def calculateLoss(self):
        """Loss function implementation of Equation 1"""
        loss = 0
        for i in range(len(self.cluster_centers)):
            for j in range(len(self.clusters[i])):
                distance = Distance.calculateMinkowskiDistance(self.clusters[i][j], self.cluster_centers[i], 2)
                loss += distance ** 2
        return loss

    def run(self):
        """Kmeans++ algorithm implementation"""
        # Choose one center uniformly at random among the data points.
        initial_centers = np.random.choice(self.dataset.shape[0], 1, replace=False)
        self.cluster_centers = {0: self.dataset[initial_centers[0]]}
        # For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
        Dx = []
        for i in range(self.dataset.shape[0]):
            min_dist = float('inf')
            for j in range(len(self.cluster_centers)):
                dist = Distance.calculateMinkowskiDistance(self.dataset[i], self.cluster_centers[j], 2)
                if dist < min_dist:
                    min_dist = dist
            Dx.append(min_dist ** 2)
        # Choose one new data point at random as a new center, using a weighted probability distribution where a point
        # x is chosen with probability proportional to D(x)^2.
        for i in range(self.K - 1):
            sumDx = sum(Dx)
            prob = [x / sumDx for x in Dx]
            new_center = np.random.choice(self.dataset.shape[0], 1, replace=False, p=prob)
            self.cluster_centers[i + 1] = self.dataset[new_center[0]]

            for j in range(self.dataset.shape[0]):
                min_dist = float('inf')
                for k in range(len(self.cluster_centers)):
                    dist = Distance.calculateMinkowskiDistance(self.dataset[j], self.cluster_centers[k], 2)
                    if dist < min_dist:
                        min_dist = dist
                Dx[j] = min_dist ** 2

        # create empty dictionary to store the previous cluster centers
        previous_centers = {i: [0] for i in range(self.K)}

        # run the Kmeans algorithm until convergence
        dict1 = None

        while not self.compareDict(self.cluster_centers, previous_centers):
            previous_centers = copy.deepcopy(self.cluster_centers)
            # assign each data point to the closest cluster center
            for i in range(self.dataset.shape[0]):
                min_dist = float('inf')
                min_cluster = -1
                for j in range(self.K):
                    dist = Distance.calculateMinkowskiDistance(self.dataset[i], self.cluster_centers[j], 2)
                    if dist < min_dist:
                        min_dist = dist
                        min_cluster = j
                self.clusters[min_cluster].append(self.dataset[i])

            dict1 = copy.deepcopy(self.clusters)
            # update the cluster centers
            for i in range(self.K):
                if len(self.clusters[i]) != 0:
                    self.cluster_centers[i] = np.mean(self.clusters[i], axis=0)
                self.clusters[i] = []

        self.clusters = dict1
        return self.cluster_centers, self.clusters, self.calculateLoss()
