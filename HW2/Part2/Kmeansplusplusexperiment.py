import numpy as np

from Part2.KMeansPlusPlus import KMeansPlusPlus
import pickle
import matplotlib.pyplot as plt

dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))

# for k value 2 to 10 for dataset1 and dataset2 run KMeans for 10 times and calculate the loss
loss_scores_dataset1 = []
loss_scores_dataset2 = []

for k in range(2, 11):
    loss_scores_dataset1.append([])
    loss_scores_dataset2.append([])

    for i in range(10):
        lowest_loss1 = float('inf')
        lowest_loss2 = float('inf')

        for j in range(10):
            kmeans = KMeansPlusPlus(dataset1, k)
            kmeans2 = KMeansPlusPlus(dataset2, k)
            cluster_centers1, clusters1, loss1 = kmeans.run()
            cluster_centers2, clusters2, loss2 = kmeans2.run()

            if loss1 < lowest_loss1:
                lowest_loss1 = loss1

            if loss2 < lowest_loss2:
                lowest_loss2 = loss2

        loss_scores_dataset1[k-2].append(lowest_loss1)
        loss_scores_dataset2[k-2].append(lowest_loss2)

# means of the loss scores for each k value
mean_loss_scores_dataset1 = []
mean_loss_scores_dataset2 = []

for i in range(len(loss_scores_dataset1)):
    mean_loss_scores_dataset1.append(np.mean(loss_scores_dataset1[i]))
    mean_loss_scores_dataset2.append(np.mean(loss_scores_dataset2[i]))

# calculate confidence interval of loss_scores_dataset1 and loss_scores_dataset2
for i in range(2, 11):
    # calculate confidence interval for dataset1
    mean1 = np.mean(loss_scores_dataset1[i-2])
    std1 = np.std(loss_scores_dataset1[i-2])

    mean2 = np.mean(loss_scores_dataset2[i-2])
    std2 = np.std(loss_scores_dataset2[i-2])

    print("For k = " + str(i) + " the mean loss for dataset1 is " + str(mean1) + " and the confidence interval is " + str(mean1 - 1.96 * std1) + " to " + str(mean1 + 1.96 * std1))
    print("For k = " + str(i) + " the mean loss for dataset2 is " + str(mean2) + " and the confidence interval is " + str(mean2 - 1.96 * std2) + " to " + str(mean2 + 1.96 * std2))

# plot the mean loss scores for dataset1 and dataset2
plt.plot(range(2, 11), mean_loss_scores_dataset1)
plt.xlabel("k")
plt.ylabel("average loss")
plt.title("Kmeans++ Average loss for dataset1")
plt.show()

plt.plot(range(2, 11), mean_loss_scores_dataset2)
plt.xlabel("k")
plt.ylabel("average loss")
plt.title("Kmeans++ Average loss for dataset2")
plt.show()
