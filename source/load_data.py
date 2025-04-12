# basic
import os
from pathlib import Path
import pandas as pd
import numpy as np

# NLP
import spacy

# case-specific
import ijson
import random
from colorama import Fore, init, Style

"""
File for loading data, tagging with spacy and feeding word2vec model.
"""


def prepare_data_frame(input_path: str, nlp: spacy.Language, chunksize: int = 10):


    # We take .json file, load it, preprocess and work on that later
    assert os.path.exists(input_path)

    # prepare output patdh
    if input_path.endswith('.json'):

        folder_path = 'files\dfs'
        input_path = Path(input_path)
        input_file_name = input_path.name
        print(f'Loading data from {input_file_name}')
        output_file_name = input_file_name.replace('.json', '')

        # for naming convention
        name_number = 0
        rows = []

        # open large json file
        with open(input_path, "rb") as f:
            obj = ijson.items(f, "documents.list.item")

            print(f"Tagging {input_file_name} text with spacy has started.")

            # TODO: fixed here

            texts = []
            metadata = []
            for _, record in enumerate(obj):
                content = record["content"]
                text = content[1]['values'][0]
                date = content[2]['values'][0]
                semantic_id = content[0]['values'][0]
                # mock metadata
                colors = ['psychology', 'ethics', 'philosphy']
                group = random.choice(colors)

                texts.append(text)
                print(len(texts))
                metadata.append((date, semantic_id, group))
                print(metadata)

                if len(texts) < chunksize:
                    docs = nlp.pipe(texts, batch_size=10, disable=["ner", "parser"])
                    for doc, (date, semantic_id, group) in zip(docs, metadata):
                        clean_text = tag_with_spacy(doc)
                        print(clean_text[0])
                        print(clean_text[1])
                        for i in enumerate(clean_text[0]):
                            if len(clean_text[0][i]) > 5:
                                rows.append({
                                    "clean_text": clean_text[0][i],
                                    "text": clean_text[1][i],
                                    "date": date,
                                    "semantic_id": semantic_id,
                                    'group': group})
                    df = pd.DataFrame(rows)
                    with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 500):
                        print(df)
                        print(df.info())


                if len(texts) >= chunksize:
                    output_file_name = output_file_name + f'{name_number}_prp.pkl'
                    output_path = os.path.join(folder_path, output_file_name)
                    print(f"Tagging done. Saving the file to {output_path} ")
                    print("-"*50)

                    # safe the chunk
                    # df.to_pickle(output_path)
                    with pd.option_context("display.max_colwidth", None):
                        print(df)

                    name_number += 1


def tag_with_spacy(doc):

    # keep clean text
    clean_tokens = []
    tokens = []

    # tag and clean
    clean_par = []
    normal_par = []


    for token in doc:
        # last token
        if token.i == len(doc) - 1:
            clean_tokens.append(clean_par)
            tokens.append(normal_par)

        # normal token
        elif token.is_space and '\n' in token.text:
            clean_tokens.append(clean_par)
            tokens.append(normal_par)
            normal_par.append(token)
            clean_par = []

        # new paragraph
        elif not (
                token.is_stop or
                token.is_punct or
                token.is_space or
                token.like_url or not
                token.is_alpha or
                len(token.text) <= 2):
            clean_par.append(token.lemma_)

        else:
            normal_par.append(token)

    return clean_tokens, tokens


def load_data(dir_with_corpus_files: str, nlp: spacy.Language, chunksize: int = 10):

    path = Path(dir_with_corpus_files)
    assert os.path.exists(path), f"Provided path {dir_with_corpus_files} does not exist"
    assert os.path.isdir(path), f"Provided path {dir_with_corpus_files} is not dir path!"

    for file in os.listdir(dir_with_corpus_files):
        filename = os.fsdecode(file)
        if filename.endswith(".json"):
            directory = os.path.join(dir_with_corpus_files, filename)
            prepare_data_frame(directory, nlp, chunksize)
        else:
            print(f"Provided file {filename} is not json. Skipping the file...")


# Generator for feeding word2vec model
class TxtSubdirsCorpus(object):
    def __init__(self, pd_path):
        print("Corpus active! Path:", pd_path, "\n")
        self.top_dir = pd_path

    def __iter__(self):
        for tokens in self.iter_documents():
            yield tokens

    def iter_documents(self):
        for root, _, files in os.walk(self.top_dir):
            for file in files:
                if file.endswith(".pkl"):
                    file_path = os.path.join(root, file)
                    df = pd.read_pickle(file_path)
                    for _, row in df.iterrows():
                        tokens = row.get("clean_text", "")
                        if not tokens:
                            init()
                            print(Fore.YELLOW + F"Warning: empty or missing 'clean_text' in a row {tokens} ")
                            print(Style.RESET_ALL)
                        else:
                            yield tokens