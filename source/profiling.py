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

# profling
import time
from memory_profiler import profile

# change working directory
os.chdir("C:\BA_thesis\BA_v2_31.03")

print(f"working directory: {os.getcwd()}")
input_path = os.getcwd() + '/files/corpus_data'

@profile
def main():
    nlp = spacy.load("en_core_web_sm")
    ld.load_data(dir_with_corpus_files=input_path,
                 nlp=nlp)


if __name__ == '__main__':
    import cProfile
    import pstats
    import io

    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()

    s = io.StringIO()
    sortby = 'cumtime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30)
    print(s.getvalue())