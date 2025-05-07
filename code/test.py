import os
import pandas as pd
from code.preprocessing import load_data as ld
import spacy
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
input_path = os.getcwd() + '/files/mock'
print(input_path)


nlp = spacy.load("en_core_web_md")
ld.load_data(dir_with_corpus_files=input_path,
             nlp=nlp)
