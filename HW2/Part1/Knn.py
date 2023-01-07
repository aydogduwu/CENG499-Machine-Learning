import numpy as np


class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters

    def majorityVote(self, neighbors):
        """Majority vote implementation"""
        return np.argmax(np.bincount(neighbors))

    def predict(self, instance):
        """KNN prediction implementation"""
        # remove instance from dataset and labels
        # dataset = np.delete(self.dataset, np.where((self.dataset == instance).all(axis=1)), axis=0)
        # labels = np.delete(self.dataset_label, np.where((self.dataset == instance).all(axis=1)), axis=0)
        # calculate the distance between the instance and all the data points in the dataset
        distances = np.empty(self.dataset.shape[0])
        if self.similarity_function_parameters is None:
            for i in range(self.dataset.shape[0]):
                distances[i] = self.similarity_function(self.dataset[i], instance)

        else:
            for i in range(self.dataset.shape[0]):
                distances[i] = self.similarity_function(self.dataset[i], instance, self.similarity_function_parameters)

        # sort the distances and get the indices of the K nearest neighbors
        sorted_indices = np.argsort(distances)[:self.K]

        # get the labels of the K nearest neighbors
        neighbors = self.dataset_label[sorted_indices]

        # return the majority vote of the K nearest neighbors
        return self.majorityVote(neighbors)


