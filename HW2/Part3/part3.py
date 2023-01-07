import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples


def plot_dendrogram(model, **kwargs):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


dataset = pickle.load(open("../data/part3_dataset.data", "rb"))

model_K2_single_euclidean = AgglomerativeClustering(n_clusters=2, linkage='single', affinity='euclidean',
                                                    compute_distances=True).fit(dataset)

model_K2_single_cosine = AgglomerativeClustering(n_clusters=2, linkage='single', affinity='cosine',
                                                 compute_distances=True).fit(dataset)

model_K2_complete_euclidean = AgglomerativeClustering(n_clusters=2, linkage='complete', affinity='euclidean',
                                                      compute_distances=True).fit(dataset)

model_K2_complete_cosine = AgglomerativeClustering(n_clusters=2, linkage='complete', affinity='cosine',
                                                   compute_distances=True).fit(dataset)

model_K3_single_euclidean = AgglomerativeClustering(n_clusters=3, linkage='single', affinity='euclidean',
                                                    compute_distances=True).fit(dataset)

model_K3_single_cosine = AgglomerativeClustering(n_clusters=3, linkage='single', affinity='cosine',
                                                 compute_distances=True).fit(dataset)

model_K3_complete_euclidean = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity='euclidean',
                                                      compute_distances=True).fit(dataset)

model_K3_complete_cosine = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity='cosine',
                                                   compute_distances=True).fit(dataset)

model_K4_single_euclidean = AgglomerativeClustering(n_clusters=4, linkage='single', affinity='euclidean',
                                                    compute_distances=True).fit(dataset)

model_K4_single_cosine = AgglomerativeClustering(n_clusters=4, linkage='single', affinity='cosine',
                                                 compute_distances=True).fit(dataset)

model_K4_complete_euclidean = AgglomerativeClustering(n_clusters=4, linkage='complete', affinity='euclidean',
                                                      compute_distances=True).fit(dataset)

model_K4_complete_cosine = AgglomerativeClustering(n_clusters=4, linkage='complete', affinity='cosine',
                                                   compute_distances=True).fit(dataset)

model_K5_single_euclidean = AgglomerativeClustering(n_clusters=5, linkage='single', affinity='euclidean',
                                                    compute_distances=True).fit(dataset)

model_K5_single_cosine = AgglomerativeClustering(n_clusters=5, linkage='single', affinity='cosine',
                                                 compute_distances=True).fit(dataset)

model_K5_complete_euclidean = AgglomerativeClustering(n_clusters=5, linkage='complete', affinity='euclidean',
                                                      compute_distances=True).fit(dataset)

model_K5_complete_cosine = AgglomerativeClustering(n_clusters=5, linkage='complete', affinity='cosine',
                                                   compute_distances=True).fit(dataset)

models = [model_K2_single_euclidean, model_K2_single_cosine, model_K2_complete_euclidean, model_K2_complete_cosine,
          model_K3_single_euclidean, model_K3_single_cosine, model_K3_complete_euclidean, model_K3_complete_cosine,
          model_K4_single_euclidean, model_K4_single_cosine, model_K4_complete_euclidean, model_K4_complete_cosine,
          model_K5_single_euclidean, model_K5_single_cosine, model_K5_complete_euclidean, model_K5_complete_cosine]

model_names = ["K=2, Single, Euclidean", "K=2, Single, Cosine", "K=2, Complete, Euclidean", "K=2, Complete, Cosine",
               "K=3, Single, Euclidean", "K=3, Single, Cosine", "K=3, Complete, Euclidean", "K=3, Complete, Cosine",
               "K=4, Single, Euclidean", "K=4, Single, Cosine", "K=4, Complete, Euclidean", "K=4, Complete, Cosine",
               "K=5, Single, Euclidean", "K=5, Single, Cosine", "K=5, Complete, Euclidean", "K=5, Complete, Cosine"]

k_values = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]

# plot dendrograms
for i in range(len(models)):
    plt.figure(figsize=(10, 5))
    plt.title("Dendrogram for " + model_names[i])
    plot_dendrogram(models[i], truncate_mode='level', p=k_values[i])
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.savefig("dendrogram_" + str(model_names[i]) + ".png")
    plt.close()

for i in range(16):  # for each model

    # code example from scikit-learn.org

    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(9, 7)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, len(dataset) + (k_values[i] + 1) * 10])

    silhouette_avg = silhouette_score(dataset, models[i].labels_)
    print(
        "For " + str(model_names[i]),
        "The average silhouette_score is :",
        silhouette_avg,
    )
    sample_silhouette_values = silhouette_samples(dataset, models[i].labels_)

    y_lower = 10
    for j in range(k_values[i]):
        ith_cluster_silhouette_values = sample_silhouette_values[models[i].labels_ == j]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(j) / k_values[i])
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax1.text(
            -0.05,
            y_lower + 0.5 * size_cluster_i,
            str(j),
        )

        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for " + str(model_names[i]))
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("K Value")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-1, -0.5, 0, 0.2, 0.4, 0.6, 0.8, 1])
    fig.savefig("silhouette_" + str(model_names[i]) + ".png")

    