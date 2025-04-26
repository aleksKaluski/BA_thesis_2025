from fontTools.misc.bezierTools import namedtuple

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
import hdbscan
import matplotlib.pyplot as plt
import optuna

os.chdir(r"C:/BA_thesis/BA_v2_31.03")

# df = pd.read_pickle('files/dfs/religion_0_prp.pkl')
#
# model = Word2Vec.load('files/models/w3e127sg1v115_best.model')
#
# # add vector represenation to each text
# dm.add_document_vector(df, model)
#
# # extract the dimentions for reduction
# vec = dm.x_from_df(df, 'vector')
# print(vec.head())
#
# # reduce the dimentions
# df = dm.reduce_dimentionality_umap(df_vectors=vec,
#                               df_normal=df,
#                               rdims=2)

# df = pd.read_pickle('files/df_to_viz')
# data = df[[x for x in df.columns if x.startswith('Dim ')]]
# data_2dims = dm.reduce_dims_with_PCA(data=data,
#                                      input_rdims=4,
#                                      output_rdims=2)
#
#
# best_gmm = cl.find_best_gmm(data_2dims, 3)
# model = cl.run_best_gmm(data_2dims, best_gmm)
# vs.plot_gmm(model, data_2dims)


# input_path = os.getcwd() + '/files/corpus_data'
# nlp = spacy.load("en_core_web_sm")
# ld.load_data(dir_with_corpus_files=input_path,
#              nlp=nlp)

# df = pd.read_pickle('files/dfs/meaning_0_prp.pkl')
# print(df)

