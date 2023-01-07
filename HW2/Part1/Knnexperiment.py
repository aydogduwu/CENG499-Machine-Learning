from sklearn.model_selection import StratifiedKFold
from configurations import *

# create a list of all the classifiers
knn_list = [knn_cosine_k3, knn_minkowski_k3, knn_mahalanobis_k3,
            knn_cosine_k5, knn_minkowski_k5, knn_mahalanobis_k5,
            knn_cosine_k7, knn_minkowski_k7, knn_mahalanobis_k7,
            knn_cosine_k9]

# create a list of all the classifiers' names
knn_names = ["Cosine K=3", "Minkowski K=3", "Mahalanobis K=3",
             "Cosine K=5", "Minkowski K=5", "Mahalanobis K=5",
             "Cosine K=7", "Minkowski K=7", "Mahalanobis K=7",
             "Cosine K=9"]

# StratifiedKFold instance for splitting the dataset into partitions with equal class proportions.
# The dataset is split into 10 partitions and shuffled.
skf = StratifiedKFold(n_splits=10, shuffle=True)

# create a list of all the accuracies
accuracies = []

# for each classifier in knn_list,
# calculate the average accuracy over 10-fold cross validation and each classifier will run 5 times
for i in range(len(knn_list)):
    # create a list to store the accuracy of each classifier for each iteration
    accuracy_list = []

    # run each classifier 5 times
    for j in range(5):
        # create a list to store the accuracy of each classifier for each iteration
        accuracy_list_fold = []

        # for each fold, calculate the accuracy of the classifier
        for train_index, test_index in skf.split(dataset, labels):
            # split the dataset into training and testing sets
            train_set = dataset[train_index]
            train_labels = labels[train_index]
            test_set = dataset[test_index]
            test_labels = labels[test_index]

            # change the dataset and labels of the classifier with new training set and labels
            knn_list[i].dataset = train_set
            knn_list[i].dataset_label = train_labels

            # predict the labels of the test set using apply along axis
            predicted_labels = np.apply_along_axis(knn_list[i].predict, 1, test_set)

            # calculate the accuracy of the classifier
            accuracy = (np.sum(predicted_labels == test_labels) / len(test_labels)) * 100

            # append the accuracy to the accuracy_list_fold
            accuracy_list_fold.append(accuracy)

        # calculate the average accuracy of the classifier over 10 folds
        accuracy_list.append(np.mean(accuracy_list_fold))

    # calculate the average accuracy of the classifier over 5 iterations
    accuracies.append(accuracy_list)

# compute confidence intervals of each array in accuracies
for i in range(len(accuracies)):
    print(knn_names[i], "Mean Accuracy:", np.mean(accuracies[i]).__format__('0.2f'))
    print(knn_names[i], "Confidence Interval Left:", (np.mean(accuracies[i]) - 1.96 * np.std(accuracies[i])).__format__('0.2f'),
                        "Confidence Interval Right:", (np.mean(accuracies[i]) + 1.96 * np.std(accuracies[i])).__format__('0.2f'))
    print()




