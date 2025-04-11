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


def prepare_data_frame(input_path: str, nlp: spacy.Language):


    # We take .json file, load it, preprocess and work on that later
    assert os.path.exists(input_path)

    # prepare output patdh
    if input_path.endswith('.json'):

        folder_path = 'files\dfs'
        input_path = Path(input_path)
        input_file_name = input_path.name
        print(f'\nLoading data from {input_file_name}')
        output_file_name = input_file_name.replace('.json', '')

        # for naming convention
        name_number = 0

        # prepare empty df for results
        df = pd.DataFrame(data={"clean_text": [],
                                "semantic_id": [],
                                "date": [],
                                "group": []})

        # open large json file
        with open(input_path, "rb") as f:
            obj = ijson.items(f, "documents.list.item")

            # for making the file smaller (to avoud memory error)
            chunkize = 1000000

            print(f"Tagging {input_file_name} text with spacy has started.")
            for index, record in enumerate(obj):
                # print(record)
                clean_data = []

                content = record["content"]
                text = content[1]['values']
                date = content[2]['values']
                semantic_id = content[0]['values']
                # mock metadata
                colors = ["red", "blue", "green", "yellow", "purple", "orange"]
                group = random.choice(colors)

                # tag and clean text
                clean_text = tag_with_spacy(text[0], nlp)
                for t in clean_text:
                    if len(t) > 5:
                        df2 = pd.DataFrame({
                            # for now it's a string
                            "clean_text": [t],
                            "date": date,
                            "semantic_id": semantic_id,
                            'group': group})
                        df = pd.concat([df, df2])
                    else:
                        continue

                # divide df to smaller chunks
                # Check if it's the last record using the index
                if len(df) > chunkize or index == len(records) - 1:
                    output_file_name = output_file_name + f'{name_number}_prp.pkl'
                    output_path = os.path.join(folder_path, output_file_name)
                    print(f"Tagging done. Saving the file to {output_path} ")
                    print("-"*50)

                    # safe the chunk
                    df.to_pickle(output_path)

                    #TODO: at this point it migth be necessary to convert clean data to a list
                    # df.clean_text.tolist()

                    name_number += 1

                    # reset df
                    df = pd.DataFrame(data={"clean_text": [],
                                            "semantic_id": [],
                                            "date": [],
                                            "group": []})


def tag_with_spacy(text: str, nlp: spacy.Language) -> list:
    doc = nlp(text)

    # keep clean text
    clean_tokens = []

    # tag and clean
    par = []
    for token in doc:
        # last token
        if token.i == len(doc) - 1:
            clean_tokens.append(par)

        # normal token
        elif token.is_space and '\n' in token.text:
            clean_tokens.append(par)
            par = []
        # new paragraph
        elif not (
                token.is_stop or
                token.is_punct or
                token.is_space or
                token.like_url or not
                token.is_alpha or
                len(token.text) <= 2):
            par.append(token.lemma_)

    return clean_tokens


def load_data(dir_with_corpus_files: str, nlp: spacy.Language):

    path = Path(dir_with_corpus_files)
    assert os.path.exists(path), f"Provided path {dir_with_corpus_files} does not exist"
    assert os.path.isdir(path), f"Provided path {dir_with_corpus_files} is not dir path!"

    for file in os.listdir(dir_with_corpus_files):
        filename = os.fsdecode(file)
        if filename.endswith(".json"):
            directory = os.path.join(dir_with_corpus_files, filename)
            prepare_data_frame(directory, nlp)
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