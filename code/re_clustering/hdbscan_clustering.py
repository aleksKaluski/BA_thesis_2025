# basic
import pandas as pd
import numpy as np

# case-specific
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from jupyter_server.auth import passwd
from pandas.core.interchange.dataframe_protocol import DataFrame


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

    # plot cluster centers
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

    # add legend manually
    legend_patches = []
    for idx, color in enumerate(color_palette):
        legend_patches.append(mpatches.Patch(color=color, label=f'Cluster {idx}'))
    if -1 in labels:
        legend_patches.append(mpatches.Patch(color=(0.5, 0.5, 0.5), label='Noise'))
    plt.legend(handles=legend_patches, title="Clusters")

    plt.title("The notion of 'direct perception' (HDBSCAN)", fontsize=14, fontweight="bold")
    plt.xlabel("Dimention no.1", fontsize=12, fontweight="bold")
    plt.ylabel("Dimention no.2", fontsize=12, fontweight="bold")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# def print_hdb_content(df :pd.DataFrame, *arg):
    
