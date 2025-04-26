# basic
import pandas as pd
import numpy as np


# case-specific
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from itertools import product
import matplotlib.pyplot as plt



def find_best_kminibatch(data: pd.DataFrame, cluster_grid: list, batch_size_grid: list):
    results = pd.DataFrame(columns=['n_clusters', 'batch_size', 'silhouette_score'])
    for n, b in product(cluster_grid, batch_size_grid):
        kmeans = MiniBatchKMeans(n_clusters=n,
                                 random_state=0,
                                 batch_size=b,
                                 n_init="auto")

        kmeans.fit(data)
        k_labels = kmeans.labels_
        silhouette_score = metrics.silhouette_score(data, k_labels, metric='euclidean')
        results.loc[len(results)] = [n, b, silhouette_score]
    results.sort_values(by=['silhouette_score'], ascending=False)
    return results.iloc[0]


def plot_kminibatch(data: list, n_clusters: int, batch_size: int, rdims: int = 2, colors: list = None):
    if colors is None:
        colors = []
    if rdims > 2:
        # reduce dimentions for visualization
        pca = PCA(n_components=2)
        x_principal = pca.fit_transform(data)
        x_principal = pd.DataFrame(x_principal, columns=['Dim 1', 'Dim 2'])
        title = "MiniBatchKMeans results after PCA reduction"

    elif rdims == 2:
        x_principal = data
        title = "MiniBatchKMeans clustering results"
    else:
        raise ValueError("rdims must be 2 or more!")

    n_clusters = int(n_clusters)
    batch_size = int(batch_size)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                             random_state=0,
                             batch_size=batch_size,
                             n_init="auto")

    kmeans.fit(x_principal)
    x_principal = x_principal.to_numpy()

    # centroids
    k_centers = kmeans.cluster_centers_

    # lables
    k_labels = kmeans.labels_

    # generate colors randomly
    if colors == []:
        colors = [np.random.rand(3, ) for _ in range(n_clusters)]
        palette = colors
    else:
        assert len(colors) == n_clusters, f"len(colors) {len(colors)} != {n_clusters} n_clusters"
        palette = colors

    plt.figure(figsize=(8, 6))
    for cluster_id in range(n_clusters):
        # pick centroid
        cluster_center = k_centers[cluster_id]
        color = palette[cluster_id]

        # plot cluster points
        cluster_points = x_principal[k_labels == cluster_id]

        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f"Cluster {cluster_id}",
            marker="o",
            color=color,
            alpha=0.7,
            edgecolors='k',
            s=30
        )
        plt.scatter(cluster_center[0],
                    cluster_center[1],
                    marker="o",
                    c=color,
                    alpha=1,
                    edgecolors='red',
                    linewidths=1,
                    s=70)
    plt.legend(loc="best", fontsize=9, frameon=True, title="Clusters")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Dimention no.1", fontsize=12, fontweight="bold")
    plt.ylabel("Dimention no.2", fontsize=12, fontweight="bold")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
