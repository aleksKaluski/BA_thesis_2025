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
    print(f"Model [{model}] was evaluated:")
    print(f"Custom similarity score: {custom_sim_score}")
    print(f"External accuracy: {acc}")
    return custom_sim_score, acc
