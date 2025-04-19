# basic
import os
from pathlib import Path
import pandas as pd
import numpy as np

# case-specific
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import MiniBatchKMeans
from matplotlib.colors import LogNorm
from wasabi import color


def plot_kminibatch(data: list, n_clusters: int, batch_size: int, colors: list = None):
    if colors:
        assert len(colors) == n_clusters, F"len(colors) {len(colors)} != {n_clusters} n_clusters"

    n_clusters = int(n_clusters)
    batch_size = int(batch_size)

    # reduce dimentions for visualization
    pca = PCA(n_components=2)
    x_principal = pca.fit_transform(data)
    x_principal = pd.DataFrame(x_principal, columns=['P1', 'P2'])

    # convert to numpy
    x_principal = x_principal.to_numpy()

    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                             random_state=0,
                             batch_size=batch_size,
                             n_init="auto")

    kmeans.fit(x_principal)

    # centroids
    k_centers = kmeans.cluster_centers_
    k_labels = kmeans.labels_

    # generate colors randomly
    if colors is None:
        colors = [np.random.rand(3, ) for _ in range(n_clusters)]
        palette = colors
    else:
        assert len(colors) == n_clusters
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
    plt.title(f"MiniBatchKMeans results after PCA reduction ", fontsize=14, fontweight="bold")
    plt.xlabel("Dimention no.1", fontsize=12, fontweight="bold")
    plt.ylabel("Dimention no.2", fontsize=12, fontweight="bold")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def print_w2v_evalutaion_results(df: pd.DataFrame, external_sim_score: str, internal_sim_score: str, model_name: str):
    # the best model is the first one

    ess = external_sim_score
    print(f"ess: {ess}")
    iss = internal_sim_score
    print(f"iss: {iss}")
    sns.set_theme(style="ticks")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df,
                    x=iss,
                    y=ess,
                    hue=model_name,
                    palette="hls",
                    s=100,
                    alpha=0.8,
                    legend=False)

    best_row = df.iloc[0]
    second_row = df.iloc[1]
    third_row = df.iloc[2]

    rows = [best_row, second_row, third_row]
    for row in rows:
        plt.scatter(row[iss],
                    row[ess],
                    s=90,
                    edgecolor='black',
                    facecolor='red',
                    linewidth=1)

        plt.text(row[iss] + 0.005,
                 row[ess] - 0.0005,
                 f"{row[model_name]}",
                 fontsize=9,
                 color='black')

    plt.title("Evaluation Results", fontsize=14, fontweight="bold")
    plt.xlabel("Mean similarity score for chosen word-pairs", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy score computed with Google test-set", fontsize=12, fontweight="bold")
    # plt.legend(title="Model", loc='best', prop={'size': 8})
    plt.tight_layout()
    plt.show()



def plot_dimentions(df: pd.DataFrame):
    if isinstance(df, pd.DataFrame):
        data = df[[x for x in df.columns if x.startswith('Dim ')]]
        pca = PCA(n_components=2)
        x_principal = pca.fit_transform(data)
        x_principal = pd.DataFrame(x_principal, columns=['P1', 'P2'])

        plt.figure(figsize=(7, 5))
        sns.set_theme(style="ticks")
        sns.relplot(data=df,
                    x=x_principal['P1'],
                    y=x_principal['P2'],
                    s=25,
                    alpha=0.9,
                    edgecolors='white',
                    legend="auto")


        plt.title(f"Distance of document vectors (PCA)", fontsize=12, fontweight="bold")
        plt.xlabel("Dimention no.1", fontsize=12, fontweight="bold")
        plt.ylabel("Dimention no.2", fontsize=12, fontweight="bold")
        max_len = max(x_principal["P1"].max(), x_principal["P2"].max())
        min_len = min(x_principal["P1"].min(), x_principal["P2"].min())
        plt.xlim(min_len - 4, max_len + 4)
        plt.ylim(min_len - 4, max_len + 4)
        plt.tight_layout()
        plt.show()

    elif isinstance(df, str):
        my_file = Path(df)
        if my_file.is_file():
            try:
                df = pd.read_pickle(df)
                # recusion for easy printing
                plot_dimentions(df)
            except TypeError:
                print(f"You passed a string {df} as a df, but it's not a pickle file.")
                return
        else:
            print(f"You passed a string {df} as a df, but it's neither a file, nor a dataframe.")
            return


def plot_dendrogram(model, **kwargs):
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
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, **kwargs)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()



def plot_gmm(gmm_model, data: list):

    x_principal = pd.DataFrame(data, columns=['P1', 'P2'])

    #TODO: fix the size of the plot
    x = np.linspace(-20.0, 30.0)
    y = np.linspace(-20.0, 40.0)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -gmm_model.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(
        X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
    )
    CB = plt.colorbar(CS, shrink=0.8, extend="both")
    plt.scatter(x_principal["P1"], x_principal["P2"], s=0.8)

    plt.title("Negative log-likelihood predicted by a GMM")
    plt.axis("tight")
    plt.show()