from gensim.utils import chunkize

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
input_path = os.getcwd() + r'\files\mock_corpus_data'
print(input_path)

# @profile
def main():
    nlp = spacy.load("en_core_web_sm")
    ld.load_data(dir_with_corpus_files=input_path,
                 nlp=nlp,
                 chunksize=2)

main()

#
# if __name__ == '__main__':
#     # import cProfile
#     # import pstats
#     # import io
#     #
#     # pr = cProfile.Profile()
#     # pr.enable()
#     # main()
#     os.chdir("C:\BA_thesis\BA_v2_31.03")
#     df = 'files/dfs/mock_meaning0_prp.pkl'
#     with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 500):
#         print(df)
#     # pr.disable()
#     #
#     # s = io.StringIO()
#     # sortby = 'cumtime'
#     # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#     # ps.print_stats(30)
#     # print(s.getvalue())