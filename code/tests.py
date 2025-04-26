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

df = mf.merge_df('files/dfs')
print(df.head())
print((df.info()))
df = ld.clean_df(dataframe=df,
                 column_name='text',
                 phraze='language ')

print(df.head())
print((df.info()))
