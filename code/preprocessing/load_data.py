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


def tag_with_spacy(doc):
    clean_tokens = []
    clean_par = []

    original_tokens = []
    original_par = []

    for token in doc:
        if token.is_space and '\n' in token.text:
            if clean_par:
                clean_tokens.append(clean_par)
                original_tokens.append(' '.join([t.text for t in original_par]))
            clean_par = []
            original_par = []
        elif not (
                token.is_stop or
                token.is_punct or
                token.is_space or
                token.like_url or
                not token.is_alpha or
                len(token.text) <= 2
        ):
            clean_par.append(token.lemma_)
            original_par.append(token)
        else:
            original_par.append(token)

    # last paraghraph
    if clean_par:
        clean_tokens.append(clean_par)
        original_tokens.append(' '.join([t.text for t in original_par]))

    return clean_tokens, original_tokens


def prepare_data_frame(input_path: str,
                       nlp: spacy.Language,
                       folder_path: str = 'files/dfs',
                       chunkzise: int = 100):

    input_path = Path(input_path)
    assert input_path.exists(), f"Path {input_path} does not exist"
    assert input_path.is_file(), f"Path {input_path} is not a file"

    if input_path.name.endswith('.json'):


        output_file_name = input_path.name.replace('.json', '')

        rows = []
        texts = []
        metadata = []
        name_number = 0

        with open(input_path, "rb") as f:
            obj = ijson.items(f, "documents.list.item")

            print(f"Tagging {input_path.name} text with spacy has started.")

            for record in obj:
                content = record["content"]
                text = content[1]['values'][0]
                date = content[2]['values'][0]
                semantic_id = content[0]['values'][0]
                group = random.choice(['psychology', 'ethics', 'philosophy'])

                texts.append(text)
                metadata.append((date, semantic_id, group))
                if len(texts) >= chunkzise:
                    docs = nlp.pipe(texts, batch_size=10, disable=["ner", "parser", "textcat"])

                    # zip for interation on docs and meta
                    for doc, (date, semantic_id, group) in zip(docs, metadata):
                        clean_text, original_text = tag_with_spacy(doc)
                        for i in range(len(clean_text)):
                            if len(clean_text[i]) > 5:
                                rows.append({
                                    "clean_text": clean_text[i],
                                    "text": original_text[i],
                                    "date": date,
                                    "semantic_id": semantic_id,
                                    "group": group
                                })
                    texts = []
                    metadata = []

            if texts:
                docs = nlp.pipe(texts, batch_size=10, disable=["ner", "parser", "textcat"])
                for doc, (date, semantic_id, group) in zip(docs, metadata):
                    clean_text, original_text = tag_with_spacy(doc)
                    for i in range(len(clean_text)):
                        if len(clean_text[i]) > 5:
                            rows.append({
                                "clean_text": clean_text[i],
                                "text": original_text[i],
                                "date": date,
                                "semantic_id": semantic_id,
                                "group": group
                            })


            df = pd.DataFrame(rows)
            output_file_name = output_file_name + f'_{name_number}_prp.pkl'
            output_path = os.path.join(folder_path, output_file_name)
            print(f"Tagging done. Saving the file to {output_path}")
            print("-" * 50)
            df.to_pickle(output_path)



def load_data(dir_with_corpus_files: str, nlp: spacy.Language):
    path = Path(dir_with_corpus_files)

    assert path.exists(), f"Path {dir_with_corpus_files} does not exist"
    assert path.is_dir(), f"Path {dir_with_corpus_files} is not a directory"
    print(f'\nLoading data from {path.name}')
    for file in os.listdir(dir_with_corpus_files):
        filename = os.fsdecode(file)
        if filename.endswith(".json"):
            directory = os.path.join(dir_with_corpus_files, filename)
            prepare_data_frame(directory, nlp)
        else:
            print(f"Provided file {filename} is not a json file. Skipping...")






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
