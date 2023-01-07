import numpy as np

class Distance:

    @staticmethod
    def calculateCosineDistance(x, y):
        """Cosine distance implementation"""
        cosine_distance = 1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
        return cosine_distance

    @staticmethod
    def calculateMinkowskiDistance(x, y, p=2):
        """Minkowski distance implementation"""
        minkowski_distance = np.power(np.sum(np.power(np.abs(x - y), p)), 1 / p)
        return minkowski_distance

    @staticmethod
    def calculateMahalanobisDistance(x, y, S_minus_1):
        """Mahalanobis distance implementation"""
        mahalanobis_distance = np.sqrt(np.dot(np.dot((x - y).T, S_minus_1), (x - y)))
        return mahalanobis_distance
