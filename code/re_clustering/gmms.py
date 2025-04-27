# basic
import pandas as pd
import numpy as np

# case-specific
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import LogNorm


def find_best_gmm(data: pd.DataFrame, n_components: int) -> pd.DataFrame:
    def gmm_bic_score(estimator, X):
        return -estimator.bic(X)

    print('\n' + "=" * 60)
    print("Starting Grid Search for GMM")
    print(f"Searching number of components from 2 to {n_components + 5}...")

    param_grid = {
        "n_components": range(2, n_components + 5),
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

    print("Top 5 GMM configurations sorted by BIC score:")
    print(res.sort_values(by="BIC score").head().to_string(index=False))

    return res.iloc[0]


def run_best_gmm(data: pd.DataFrame, gmm_params: pd.DataFrame):
    n_gmm = gmm_params["Number of components"]
    cov_gmm = gmm_params["Type of covariance"]
    i_gmm = gmm_params["Number of iterations"]

    print(f'\nRunning the best GMM model for:')
    print(f'number of components = {n_gmm}')
    print(f'type of covariance = {cov_gmm}')
    print(f'number of iterations = {i_gmm}')

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
    return gmm.fit(data)


def plot_gmm(data: pd.DataFrame, gmm_model: GaussianMixture):
    x_principal = pd.DataFrame(data, columns=['Dim 1', 'Dim 2'])
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
    plt.scatter(x_principal["Dim 1"], x_principal["Dim 2"], s=0.8)

    plt.title("Negative log-likelihood predicted by a GMM")
    plt.axis("tight")
    plt.show()