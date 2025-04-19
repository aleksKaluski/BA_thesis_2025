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

print(f"working directory: {os.getcwd()}")
input_path = os.getcwd() + '/files/corpus_data'

corpus = ld.TxtSubdirsCorpus("files/dfs")
