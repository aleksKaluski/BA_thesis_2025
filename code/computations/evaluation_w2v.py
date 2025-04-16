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

"""
Evaluation and piccking the best word2v3c model
"""

def evaluate_model(dir_with_models: str, ev_file: str, test_words: list) -> pd.DataFrame:
    # test words is mean to be a list of tuples
    # eg. [("game", "theory"), ("social", "concern")]

    # folder with various models
    dirpath = list(Path(dir_with_models).glob("*.model"))
    # print(dirpath)

    model_list = []
    model_filenames = []

    # prepare list with models' paths
    for model_path in dirpath:
        file_path = str(model_path)
        # print(f"Loading model: {file_path}")

        model_list.append(file_path)
        model_filenames.append(model_path.name)
        # print(model_list)

    df = pd.DataFrame(columns=['model', 'accuracy', 'similarity_score'])
    for i in range(len(model_list)):
        model = Word2Vec.load(model_list[i])
        acc = evaluation_with_file(model=model, ev_file=ev_file)
        similarity_scores = []
        for x in range(len(test_words)):
            w1 = test_words[x][0]
            w2 = test_words[x][1]
            # print(f"Evaluating {w1} and {w2} for model {model}")

            if w1 in model.wv.key_to_index and w2 in model.wv.key_to_index:
                similarity_score = model.wv.similarity(*test_words[x])
                # print(model_filenames[i], f"{w1}:{w2}", similarity_score)
                similarity_scores.append(similarity_score)
            else:
                print(f"Warning: Word {w1} or {w2} not found in the model vocabulary.")

        df.loc[len(df)] = [model_filenames[i], acc, statistics.mean(similarity_scores)]
    return df


def evaluation_with_file(model, ev_file: str):
    file_path = Path(ev_file)
    assert file_path.is_file(), f"You passed wrong file [{file_path}] in evaluation_with_file"
    result = model.wv.evaluate_word_analogies(ev_file)
    return result[0]  # accuracy


def pick_the_best_model(df: pd.DataFrame):
    vs.print_evalutaion_results(df)
    df.sort_values(by=['accuracy', 'similarity_score'], ascending=False, inplace=True)
    best_model = df.iloc[0]

    print(f"Best model: {best_model['model']}")
    print(f"Cosine Similarity: {best_model['similarity_score']}")
    print(f"Accuracy: {best_model['accuracy']}")
    return best_model


def compleate_evaluation(dir_with_models: str, ev_file: str, test_words: list):
    print(f'\nEvaluation of models has been started...')
    e = evaluate_model(dir_with_models=dir_with_models,
                       ev_file=ev_file,
                       test_words=test_words)
    return pick_the_best_model(e)


"""
Ta część to przygotowuje odfiltrowany plik Google, zgodnie z Pana pomysłem. Wiem, że jest mało wydajna,
ale cała operacja jest jednorazowa i zajmuje mniej niż minutę. Na razie z niej nie korzystam.
"""


def build_freq_dict(top_dir):
    word_counts = defaultdict(int)
    total_words = 0

    for root, _, files in os.walk(top_dir):
        for file in files:
            if file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                df = pd.read_pickle(file_path)

                for _, row in df.iterrows():
                    tokens = row.get("clean_text", [])
                    for token in tokens:
                        # print(token)
                        total_words += 1
                        word_counts[token] += 1
                        # print(word_counts)
    # Normalize
    for word in word_counts:
        word_counts[word] /= total_words
    print(word_counts)
    return word_counts


def clean_google_data(top_dir, input_file, nlp):
    print("\nGoogle data cleaning...")

    # Count word frequencies in corpus
    frequencies = build_freq_dict(top_dir)

    with open(input_file, "rb") as f, open('../files/clean_google.txt', "w", encoding="utf-8") as g:
        for line in f:
            line_str = line.decode('utf-8').strip()
            if line_str.startswith(':'):
                g.write(line_str + '\n')
            else:
                tagged_tokens = ld.tag_with_spacy(line_str, nlp)
                # If any token is frequent, write the line
                if any(frequencies.get(token, 0) > 0 for token in tagged_tokens):
                    # print(f"writing: {line_str}")
                    g.write(line_str + '\n')
