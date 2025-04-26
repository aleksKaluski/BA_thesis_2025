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

# required packages: pip install spacy pandas numpy ijson colorama matplotlib seaborn gensim umap-learn tqdm wordcloud scikit-learn hdbscan
# python -m spacy download en_core_web_sm

def main():
    """
    1) load the data
    """
    input_path = os.getcwd() + '/files/corpus_data'

    nlp = spacy.load("en_core_web_md")
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
                                        n_trials=30)

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
        vector_size=vector_size
    )
    model.save(f"files/models/w{window}e{epochs}sg{sg}v{vector_size}_best.model")

    """
    4) Reduce dimentions
    """
    # model = Word2Vec.load('files/models/w3e127sg1v115_best.model')
    df = mf.merge_df('files/dfs')

    # save df
    df.to_pickle('files/checkpoints/merged_df.pkl')

    # now we reduce data!
    df_clean = ld.clean_df(dataframe=df,
                           column_name='text',
                           phraze='language ')

    # add vector represenation to each text
    dm.add_document_vector(df_clean, model)

    # extract the dimentions for reduction
    vec = dm.x_from_df(df_clean, 'vector')

    # reduce the dimentions to 2
    df_reduced = dm.reduce_dimentionality_umap(df_vectors=vec,
                                               df_normal=df_clean,
                                               rdims=2)
    df_reduced.to_pickle('files/checkpoints/rediuced_df.pkl')

    """
    5) visualize the document distance
    """
    data = df_reduced[[x for x in df_reduced.columns if x.startswith('Dim ')]]
    wd.plot_dimentions(data=data,
                       rdims=2)

    """
    6) cluster the documents
    """

    """
    6.1) Perform mini-batch for finding the right number of clusters
    """
    best_kminibatch = mk.find_best_kminibatch(data=data,
                                              cluster_grid=[2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                              batch_size_grid=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])

    best_kminibatch.to_pickle('files/checkpoints/best_kminibatch.pkl')




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

    df_reduced['hdbscan_labels'] = hdb.labels_
    df_reduced['hdbscan_probabilities'] = hdb.probabilities_
    df_reduced.sort_values(by=['hdbscan_probabilities'], ascending=False, inplace=True)

    top = df_reduced['hdbscan_probabilities'].nlargest(n=20).index
    top = df_reduced.loc[top]
    with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 700):
        print(top)

    """
    8) Plot wordclouds for each cluster
    """
    wd.divide_and_plot(df_reduced, "hdbscan_labels")



if __name__ == '__main__':
    main()
