from code.visual import visualization as vs, word_clouds as wd
from code.preprocessing import load_data as ld, merge_df as mf
from code.computations import evaluation_w2v as ev, clustering as cl, dim_reduction as dm

# basic
import os
import pandas as pd

# case_specific
import spacy
from sklearn.cluster import AgglomerativeClustering
from itertools import product
from gensim.models import Word2Vec

# required packages: pip install spacy pandas numpy ijson colorama matplotlib seaborn gensim umap-learn tqdm wordcloud scikit-learn
# python -m spacy download en_core_web_sm

import time
from memory_profiler import profile

#TODO:
# - znajdź cenrroidy
# - dodaj gensim bigrams - nope
# - popraw wykresy
# - merytoryczny upgrade

def main():
    timings = {}

    os.chdir(r"C:/BA_thesis/BA_v2_31.03")

    print(f"working directory: {os.getcwd()}")
    input_path = os.getcwd() + '/files/corpus_data'


    """ 
    1) load the data
    """
    start = time.perf_counter()
    nlp = spacy.load("en_core_web_sm")
    ld.load_data(dir_with_corpus_files=input_path,
                 nlp=nlp,
                 chunksize=40)
    timings['load_data'] = time.perf_counter() - start


    """
    2) create corpus
    """
    start = time.perf_counter()
    corpus = ld.TxtSubdirsCorpus("files/dfs")
    timings['create_corpus'] = time.perf_counter() - start


    """
    3) Train a few models
    """

    start = time.perf_counter()
    # Parameter grid
    window = [1, 3]
    epochs = [100]
    sg = [0, 1]
    vector_size = [100]

    for w, e, s, v in product(window, epochs, sg, vector_size):
        print(f'Training of w{w}e{e}sg{s}v{v} haas started.')
        model = Word2Vec(
            sentences=corpus,
            window=w,
            min_count=5,
            epochs=e,
            sg=s,
            vector_size=v
        )
        model.save(f"files/models/w{w}e{e}sg{s}v{v}.model")

    timings['train_models'] = time.perf_counter() - start

    """
    4) Evaluate models
    """
    start = time.perf_counter()
    m = ev.compleate_evaluation(dir_with_models='files/models',
                                ev_file='files/google.txt',
                                test_words=[('good', 'bad'), ('game', 'theory')])
    timings['evaluate_models'] = time.perf_counter() - start

    """
    5) Reduce dimentions
    """
    # take the best model
    start = time.perf_counter()
    model_name = m['model']
    best_model = f'files/models/{model_name}'
    model = Word2Vec.load(best_model)

    df = mf.merge_df('files/dfs')

    # add vector represenation to each text
    dm.add_document_vector(df, model)

    # extract the dimentions for reduction
    vec = dm.x_from_df(df, 'vector')


    # reduce the dimentions
    df = dm.reduce_dimentionality(vec, df)

    timings['reduce_dimensions'] = time.perf_counter() - start


    """
    6) visualize the document distance
    """

    start = time.perf_counter()
    # df = pd.read_pickle("files/df_red.pkl")
    vs.plot_dimentions(df)
    timings['plot_dimensions'] = time.perf_counter() - start


    """
    7) cluster the documents
    """
    start = time.perf_counter()
    data = df[[x for x in df.columns if x.startswith('Dim ')]]

    """
    7.1) Perform mini-batch for finding the right number of clusters
    """
    best_kminibatch = cl.find_best_kminibatch(data=data,
                                              cluster_grid=[2, 3, 4],
                                              batch_size_grid=[10, 20])

    vs.plot_kminibatch(data=data,
                       n_clusters=best_kminibatch['n_clusters'],
                       batch_size=best_kminibatch['batch_size'])

    """
    7.2) clustering with Gaussian Mixtures
    """

    n = int(best_kminibatch['n_clusters'])
    best_gmm = cl.find_best_gmm(data, n)

    df = cl.run_best_gmm(data, best_gmm, df)
    print(df.head())


    """
    7.3) Hierarchical clustering
    """

    # AgglomerativeC clustering
    ahc = AgglomerativeClustering(n_clusters=n,
                                  metric='euclidean',
                                  compute_distances=True)

    ac_clusters = ahc.fit(data)
    vs.plot_dendrogram(ac_clusters, truncate_mode="level", p=3)


    #TODO: Optional add various labels to the dataset and compare them later
    timings['clustering'] = time.perf_counter() - start

    """
    8) Plot wordclouds for each cluster
    """
    start = time.perf_counter()
    wd.divide_and_plot(df, "gmm_labels")
    timings['wordclouds'] = time.perf_counter() - start


    with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 500):
        print(df)

    print("\nTIME REPORT")
    for step, seconds in timings.items():
        print(f"{step:>20}: {seconds:.2f} s")

if __name__ == '__main__':
    main()
