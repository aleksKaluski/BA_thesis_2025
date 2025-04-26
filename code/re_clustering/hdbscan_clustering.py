# basic
import pandas as pd
import numpy as np

# case-specific
import matplotlib.pyplot as plt
import seaborn as sns


def plot_hdbscan_points(data: pd.DataFrame, prediction_on_data):

    # Convert centers to a NumPy array for proper slicing
    centers = prediction_on_data.exemplars_

    # extract labels
    labels = prediction_on_data.labels_

    # find the number of cluster
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # set colors
    color_palette = [np.random.rand(3, ) for _ in range(n_clusters)]
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in prediction_on_data.labels_]

    cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, prediction_on_data.probabilities_)]

    plt.figure(figsize=(8, 6))
    plt.scatter(data['Dim 1'],
                data['Dim 2'],
                label=f"Cluster",
                marker="o",
                color=cluster_member_colors,
                alpha=0.5,
                edgecolors='k',
                s=30
                )

    for i, cluster in enumerate(centers):
        color = color_palette[i]
        plt.scatter(cluster[:, 0],
                    cluster[:, 1],
                    marker="o",
                    color=color,
                    alpha=0.85,
                    edgecolors='red',
                    linewidths=0.3,
                    s=40)
        i += 1
    plt.title("title", fontsize=14, fontweight="bold")
    plt.xlabel("Dimention no.1", fontsize=12, fontweight="bold")
    plt.ylabel("Dimention no.2", fontsize=12, fontweight="bold")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()