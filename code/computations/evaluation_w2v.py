# basic
import os
from pathlib import Path
import pandas as pd

# other files
from code.preprocessing import load_data as ld
from code.visual import visualization as vs

# case-specific
from gensim.models import Word2Vec
import statistics
from collections import defaultdict
import optuna

"""
Evaluation and piccking the best word2v3c model
"""


def evaluation_with_file(model: Word2Vec, ev_file: str):
    file_path = Path(ev_file)
    assert file_path.is_file(), f"You passed wrong file [{file_path}] in evaluation_with_file"
    result = model.wv.evaluate_word_analogies(ev_file)
    return result[0]  # accuracy


def evaluate_model(model: Word2Vec, ev_file: str, test_words: list):
    """
    Evaluate single word2vec model
    """
    acc = evaluation_with_file(model=model, ev_file=ev_file)
    similarity_scores = []
    for x in range(len(test_words)):
        w1 = test_words[x][0]
        w2 = test_words[x][1]

        if w1 in model.wv.key_to_index and w2 in model.wv.key_to_index:
            similarity_score = model.wv.similarity(*test_words[x])
            similarity_scores.append(similarity_score)
        else:
            print(f"Warning: Word {w1} or {w2} not found in the model vocabulary.")

    if similarity_scores:
        custom_sim_score = statistics.mean(similarity_scores)
    else:
        custom_sim_score = 0
        print(f"No similarity scores found for this model [{model}]. avg_score: {custom_sim_score} ")
    return custom_sim_score, acc


def find_best_params_w2v(corpus, n_trials: int):
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
        result = evaluate_model(w2v,
                                ev_file='files/google.txt',
                                test_words=[('good', 'bad'), ('game', 'theory')])
        return result

    study = optuna.create_study(directions=["maximize", "maximize"],
                                study_name='w2v_optimization')

    study.optimize(train_w2v_models,
                   n_trials=n_trials)

    ev_metric = study.trials_dataframe()
    ev_metric.rename(columns={'values_0': 'custom_sim_score', 'values_1': 'external_accuracy'}, inplace=True)
    ev_metric['model_name'] = (
            'w' + ev_metric['params_window'].astype(str) +
            'e' + ev_metric['params_epochs'].astype(str) +
            'sg' + ev_metric['params_sg'].astype(str) +
            'v' + ev_metric['params_vector_size'].astype(str))
    return ev_metric


def get_best_params(df: pd.DataFrame, external_sim_score: str, internal_sim_score: str):
    df = df.sort_values(by=[external_sim_score, internal_sim_score], ascending=False)
    best_params = df.iloc[0]
    return best_params


