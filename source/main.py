import clustering as cl
import visualization as vs
import load_data as ld
import word_clouds as wd
import merge_df as mf
import evaluation_w2v as ev
import dim_reduction as dm

# basic
import os
from pathlib import Path
import pandas as pd
import numpy as np

# case_specific
import spacy
from sklearn.cluster import AgglomerativeClustering
from itertools import product
from gensim.models import Word2Vec

# required packages: pip install spacy pandas numpy ijson colorama matplotlib seaborn gensim umap-learn tqdm wordcloud scikit-learn
# python -m spacy download en_core_web_sm

# change working directory
os.chdir("C:\BA_thesis\BA_v2_31.03")

print(f"working directory: {os.getcwd()}")
input_path = os.getcwd() + '/files/corpus_data'


""" 
1) load the data
"""
nlp = spacy.load("en_core_web_sm")
ld.load_data(dir_with_corpus_files=input_path,
             nlp=nlp)


"""
2) create corpus
"""
corpus = ld.TxtSubdirsCorpus("files/dfs")

"""
3) Train a few models
"""
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

"""
4) Evaluate models
"""
m = ev.compleate_evaluation(dir_with_models='files/models',
                            ev_file='files/google.txt',
                            test_words=[('good', 'bad'), ('game', 'theory')])

"""
5) Reduce dimentions
"""
# take the best model
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


"""
6) visualize the document distance
"""
# df = pd.read_pickle("files/df_red.pkl")
vs.plot_dimentions(df)

"""
7) cluster the documents
"""
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


"""
8) Plot wordclouds for each cluster
"""
wd.divide_and_plot(df, "gmm_labels")

with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 500):
    print(df)
