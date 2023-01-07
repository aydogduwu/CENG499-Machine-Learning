from Knn import KNN
from Distance import Distance
import pickle
import numpy as np

dataset, labels = pickle.load(open("../data/part1_dataset.data", "rb"))
# create 10 different KNN classifiers with different similarity functions
knn_cosine_k3 = KNN(dataset, labels, Distance.calculateCosineDistance, K=3)
knn_minkowski_k3 = KNN(dataset, labels, Distance.calculateMinkowskiDistance, similarity_function_parameters=2, K=3)
knn_mahalanobis_k3 = KNN(dataset, labels, Distance.calculateMahalanobisDistance,
                         similarity_function_parameters=np.linalg.inv(np.cov(dataset.T)), K=3)

knn_cosine_k5 = KNN(dataset, labels, Distance.calculateCosineDistance, K=5)
knn_minkowski_k5 = KNN(dataset, labels, Distance.calculateMinkowskiDistance, similarity_function_parameters=2, K=5)
knn_mahalanobis_k5 = KNN(dataset, labels, Distance.calculateMahalanobisDistance,
                         similarity_function_parameters=np.linalg.inv(np.cov(dataset.T)), K=5)

knn_cosine_k7 = KNN(dataset, labels, Distance.calculateCosineDistance, K=7)
knn_minkowski_k7 = KNN(dataset, labels, Distance.calculateMinkowskiDistance, similarity_function_parameters=2, K=7)
knn_mahalanobis_k7 = KNN(dataset, labels, Distance.calculateMahalanobisDistance,
                         similarity_function_parameters=np.linalg.inv(np.cov(dataset.T)), K=7)

knn_cosine_k9 = KNN(dataset, labels, Distance.calculateCosineDistance, K=9)