from code.visual import visualization as vs, word_clouds as wd
from code.preprocessing import load_data as ld, merge_df as mf
from code.computations import evaluation_w2v as ev, clustering as cl, dim_reduction as dm

# basic
import os
import pandas as pd
import numpy as np

# case_specific
import spacy
from sklearn.cluster import AgglomerativeClustering
from itertools import product
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import hdbscan
import matplotlib.pyplot as plt

os.chdir(r"C:/BA_thesis/BA_v2_31.03")

df = pd.read_pickle('files/df_to_viz')
print(df.info)
data = df[[x for x in df.columns if x.startswith('Dim ')]]
pca = PCA(n_components=2)
x_principal = pca.fit_transform(data)

hdb = hdbscan.HDBSCAN(min_cluster_size=30, gen_min_span_tree=True)
hdb.fit(x_principal)

hdb.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                edge_alpha=0.6,
                                node_size=80,
                                edge_linewidth=2)
plt.show()
hdb.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
plt.show()