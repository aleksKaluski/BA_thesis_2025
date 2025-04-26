# basic
import os
from pathlib import Path
import pandas as pd
import numpy as np
from numba.np.math.numbers import complex_eq_impl

# other files
from code.visual import visualization as vs

# case-specific
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from itertools import product
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV


def find_best_kminibatch(data: list, cluster_grid: list, batch_size_grid: list):

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


def find_best_gmm(data: list, n_components:int) -> pd.DataFrame:
    def gmm_bic_score(estimator, X):
        return -estimator.bic(X)


    param_grid = {
        "n_components": range(2, n_components+5),
        "covariance_type": ["spherical", "tied", "diag", "full"],
        "max_iter": [100, 150, 200]
    }
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
    )
    grid_search.fit(data)

    res = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "param_covariance_type", "param_max_iter", "mean_test_score"]
    ]
    res["mean_test_score"] = -res["mean_test_score"]
    res = res.rename(
        columns={
            "param_n_components": "Number of components",
            "param_covariance_type": "Type of covariance",
            "param_max_iter": "Number of iterations",
            "mean_test_score": "BIC score",
        }
    )
    print("Resuls of grid search for GMM:")
    print(res.sort_values(by="BIC score").head().to_string(index=False))

    return res.iloc[0]


def run_best_gmm(data:list, gmm_params: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    n_gmm = gmm_params["Number of components"]
    cov_gmm = gmm_params["Type of covariance"]
    i_gmm = gmm_params["Number of iterations"]

    #TODO: possibly this reduction wiht PCA is unecessary (I made it for visualization)

    pca = PCA(n_components=2)
    x_principal = pca.fit_transform(data)

    gmm = GaussianMixture(n_components=n_gmm,
                          covariance_type=cov_gmm,
                          random_state=53,
                          tol=0.001,
                          reg_covar=1e-06,
                          max_iter=i_gmm,
                          init_params="kmeans",
                          warm_start=True,
                          verbose=1,
                          verbose_interval=20
                          )
    gmm.fit(x_principal)


    vs.plot_gmm(gmm_model=gmm, data=x_principal)
    df["gmm_labels"] = gmm.predict(x_principal)
    return df
