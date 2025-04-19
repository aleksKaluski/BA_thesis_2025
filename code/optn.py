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



ev_metricts = pd.DataFrame(columns=['model', 'accuracy', 'similarity_score'])

def train_w2v_models(trial):
    window = trial.suggest_int("window", 2, 3)
    epochs = trial.suggest_int("epochs", 100, 150)
    sg = trial.suggest_int("sg", 0, 1)
    vector_size = trial.suggest_int("vector_size", 100, 120)
    print(f'\nTraining of w{window}e{epochs}sg{sg}v{vector_size} has started.')
    w2v = Word2Vec(
        sentences=corpus,
        window=window,
        min_count=5,
        epochs=epochs,
        sg=sg,
        vector_size=vector_size
    )

    e = ev.evaluate_model(w2v,
                          ev_file='files/google.txt',
                          test_words=[('good', 'bad'), ('game', 'theory')])

    return e


study = optuna.create_study(directions=["maximize", "maximize"],
                            study_name='w2v_optimization')
study.optimize(train_w2v_models, n_trials=3)

trial = study.best_trials

print("\nAccuracy: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))