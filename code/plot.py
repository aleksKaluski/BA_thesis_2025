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

os.chdir(r"C:/BA_thesis/BA_v2_31.03")

df = pd.read_pickle('files/df_to_viz')

data = df[[x for x in df.columns if x.startswith('Dim ')]]

ahc = AgglomerativeClustering(n_clusters=4,
                              metric='euclidean',
                              compute_distances=True)

ac_clusters = ahc.fit(data)
vs.plot_dendrogram(ac_clusters, truncate_mode="level", p=3)