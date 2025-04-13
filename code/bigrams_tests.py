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
import time


from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

def main():
    timings = {}

    os.chdir(r"C:/BA_thesis/BA_v2_31.03")

    print(f"working directory: {os.getcwd()}")
    input_path = os.getcwd() + '/files/corpus_data'

    """
    2) create corpus
    """
    start = time.perf_counter()
    corpus = ld.TxtSubdirsCorpus("files/dfs")
    timings['create_corpus'] = time.perf_counter() - start

    phrase_model = Phrases(corpus, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
    corpus = phrase_model[corpus]
    start = time.perf_counter()

    # Parameter grid
    window = [1]
    epochs = [50]
    sg = [0]
    vector_size = [70]

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


main()
