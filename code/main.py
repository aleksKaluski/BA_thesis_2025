# basic
import os
import pandas as pd

# case_specific
import spacy
from sklearn.cluster import AgglomerativeClustering
from gensim.models import Word2Vec
import hdbscan
import matplotlib.pyplot as plt

from code.preprocessing import load_data as ld
from code.preprocessing import merge_df as mf
from code.computations import evaluation_w2v as ev
from code.computations import dim_reduction as dm
from code.computations import word_clouds as wd
from code.re_clustering import minibatch_k_means as mk
from code.re_clustering import gmms as gm
from code.re_clustering import agglomerative_clustering as ag
from code.re_clustering import hdbscan_clustering as hd

os.chdir(r"C:/BA_thesis/BA_v2_31.03")
print(f"working directory: {os.getcwd()}")


def main():

    """
    1) load the data
    """
    input_path = os.getcwd() + '/files/corpus_data'

    nlp = spacy.load("en_core_web_sm")
    ld.load_data(dir_with_corpus_files=input_path,
                 nlp=nlp)

    """
    2) create corpus
    """
    corpus = ld.TxtSubdirsCorpus("files/dfs")

    """
    3) Train a few models and find the best one with optuna
    """
    ev_metric = ev.find_best_params_w2v(corpus=corpus,
                                        n_trials=3)


    ev.plot_w2v_evalutaion_results(df=ev_metric,
                                   external_sim_score='external_accuracy',
                                   internal_sim_score='custom_sim_score',
                                   model_name='model_name')

    best_params = ev.get_best_params(df=ev_metric,
                                     external_sim_score='external_accuracy',
                                     internal_sim_score='custom_sim_score')

    epochs = best_params['params_epochs']
    sg = best_params['params_sg']
    vector_size = best_params['params_vector_size']
    window = best_params['params_window']

    model = Word2Vec(
        sentences=corpus,
        window=window,
        min_count=5,
        epochs=epochs,
        sg=sg,
        vector_size=vector_size,
        workers=2
    )
    model.save(f"files/models/w{window}e{epochs}sg{sg}v{vector_size}_best.model")

    """
    4) Reduce dimentions
    """
    model = Word2Vec.load('files/models/w3e127sg1v115_best.model')
    df = mf.merge_df('files/dfs')

    # add vector represenation to each text
    dm.add_document_vector(df, model)

    # extract the dimentions for reduction
    vec = dm.x_from_df(df, 'vector')

    # reduce the dimentions to 2
    df = dm.reduce_dimentionality_umap(df_vectors=vec,
                                       df_normal=df,
                                       rdims=2)

    """
    5) visualize the document distance
    """
    data = df[[x for x in df.columns if x.startswith('Dim ')]]
    wd.plot_dimentions(data=data,
                       rdims=2)

    """
    6) cluster the documents
    """

    """
    6.1) Perform mini-batch for finding the right number of clusters
    """
    best_kminibatch = mk.find_best_kminibatch(data=data,
                                              cluster_grid=[2, 3, 4],
                                              batch_size_grid=[10, 20])

    mk.plot_kminibatch(data=data,
                       n_clusters=best_kminibatch['n_clusters'],
                       batch_size=best_kminibatch['batch_size'])

    """
    7.2) clustering with Gaussian Mixtures
    """

    n = int(best_kminibatch['n_clusters'])
    best_gmm = gm.find_best_gmm(data=data,
                                n_components=n)

    gm_model = gm.run_best_gmm(data=data, gmm_params=best_gmm)
    gm.plot_gmm(data=data,
                gmm_model=gm_model)

    """
    7.3) Hierarchical clustering
    """

    # AgglomerativeC clustering
    ahc = AgglomerativeClustering(n_clusters=n,
                                  metric='euclidean',
                                  compute_distances=True)

    ac_clusters = ahc.fit(data)
    ag.plot_dendrogram(ac_clusters, truncate_mode="level", p=3)

    """
    7.4) Clustering with HDBSCAN
    """
    hdb = hdbscan.HDBSCAN(min_cluster_size=30, gen_min_span_tree=True)
    hdb.fit(data)
    hdb.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                    edge_alpha=0.6,
                                    node_size=80,
                                    edge_linewidth=2)
    plt.show()
    hdb.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    plt.show()

    hd.plot_hdbscan_points(data=data,
                           prediction_on_data=hdb)



if __name__ == '__main__':
    main()
