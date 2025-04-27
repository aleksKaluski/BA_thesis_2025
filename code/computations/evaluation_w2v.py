# basic
from pathlib import Path
import pandas as pd

# case-specific
from gensim.models import Word2Vec
import statistics
from collections import defaultdict
import optuna

import matplotlib.pyplot as plt
import seaborn as sns
import time

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
            print(f"Warning: Word '{w1}' or word '{w2}' not found in the model vocabulary.")

    if similarity_scores:
        custom_sim_score = statistics.mean(similarity_scores)
    else:
        custom_sim_score = 0
        print(f"No similarity scores found for this model [{model}]. avg_score: {custom_sim_score} ")
    return custom_sim_score, acc


def find_best_params_w2v(corpus, n_trials: int):
    assert n_trials >= 3; "Minimal number of trials is 3!"

    print("\nLooking for best w2v model!")

    def train_w2v_models(trial):
        start_time = time.time()
        window = trial.suggest_int("window", 2, 3)
        epochs = trial.suggest_int("epochs", 100, 150)
        sg = trial.suggest_int("sg", 0, 1)
        vector_size = trial.suggest_int("vector_size", 100, 120)
        model_name = f"w{window}e{epochs}sg{sg}v{vector_size}"

        print("\n" + "=" * 60)
        print(f"Starting training: {model_name}")
        print("=" * 60)

        w2v = Word2Vec(
            sentences=corpus,
            window=window,
            min_count=5,
            epochs=epochs,
            sg=sg,
            vector_size=vector_size
        )
        w2v.save(f"files/models/w{window}e{epochs}sg{sg}v{vector_size}.model")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"time of training: {elapsed_time:.2f} seconds")

        result = evaluate_model(
            w2v,
            ev_file='files/google.txt',
            test_words=[
                ("embodied", "cognition"),
                ("motor", "system"),
                ("perceptual", "system"),
                ("social", "cognition"),
                ("situated", "cognition"),
                ("extended", "mind"),
                ("affordance", "perception"),
                ("body", "environment"),
                ("action", "perception"),
                ("mirror", "neurons"),
                ("memory", "recall"),
                ("self", "regulation"),
                ("language", "processing"),
                ("motor", "skills"),
                ("phenomenological", "experience"),
                ("brain", "body"),
                ("emotion", "action"),
                ("physical", "interaction")
            ]
        )

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


def plot_w2v_evalutaion_results(df: pd.DataFrame, external_sim_score: str, internal_sim_score: str, model_name: str):
    # the best model is the first one
    df = df.sort_values(by=[external_sim_score, internal_sim_score], ascending=False)

    ess = external_sim_score
    iss = internal_sim_score
    sns.set_theme(style="ticks")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df,
                    x=iss,
                    y=ess,
                    hue=model_name,
                    palette="hls",
                    s=100,
                    alpha=0.8,
                    legend=False)

    best_row = df.iloc[0]
    second_row = df.iloc[1]
    third_row = df.iloc[2]

    rows = [best_row, second_row, third_row]
    for row in rows:
        plt.scatter(row[iss],
                    row[ess],
                    s=90,
                    edgecolor='black',
                    facecolor='red',
                    linewidth=1)

        plt.text(row[iss] + 0.005,
                 row[ess] - 0.0005,
                 f"{row[model_name]}",
                 fontsize=9,
                 color='black')

    plt.title("Evaluation Results", fontsize=14, fontweight="bold")
    plt.xlabel("Mean similarity score for chosen word-pairs", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy score computed with Google test-set", fontsize=12, fontweight="bold")
    # plt.legend(title="Model", loc='best', prop={'size': 8})
    plt.tight_layout()
    plt.show()
