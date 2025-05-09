# basic
import os
from pathlib import Path
import pandas as pd
import re

# NLP
import spacy
import time
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
                       chunkzise: int = 3):
    input_path = Path(input_path)
    assert input_path.exists(), f"Path {input_path} does not exist"
    assert input_path.is_file(), f"Path {input_path} is not a file"

    if input_path.name.endswith('.json'):
        output_file_name = input_path.name.replace('.json', '')
        rows = []
        texts = []
        metadata = []
        name_number = 0
        doc_counter = 0  # new

        with open(input_path, "rb") as f:
            obj = ijson.items(f, "documents.list.item")

            print(f"Starting tagging {input_path.name} with spaCy...")

            for record in obj:
                doc_counter += 1
                if doc_counter % 300 == 0:
                    print(f"Processed {doc_counter} documents...")

                content = record["content"]
                content_dict = {item['name']: item['values'][0] for item in content}
                text = content_dict.get('text', '')
                date = content_dict.get('date', '')
                semantic_id = content_dict.get('id', '')
                group = random.choice(['psychology', 'ethics', 'philosophy'])

                texts.append(text)
                metadata.append((date, semantic_id, group))

                if len(texts) >= chunkzise:
                    docs = nlp.pipe(texts, batch_size=200, disable=["ner", "parser", "textcat"])
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
                docs = nlp.pipe(texts, batch_size=200, disable=["ner", "parser", "textcat"])
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
            output_path = Path(output_path)

            print(f"Tagging completed: {doc_counter} documents processed.")
            print(f"Saving file to {output_path}...")
            print("-" * 50)
            df.to_pickle(output_path)


def load_data(dir_with_corpus_files: str, nlp: spacy.Language):
    path = Path(dir_with_corpus_files)

    assert path.exists(), f"Path {dir_with_corpus_files} does not exist"
    assert path.is_dir(), f"Path {dir_with_corpus_files} is not a directory"
    print(f'\nLoading data from folder: {path.name}')
    for file in os.listdir(dir_with_corpus_files):
        filename = os.fsdecode(file)
        if filename.endswith(".json"):
            directory = os.path.join(dir_with_corpus_files, filename)
            prepare_data_frame(directory, nlp)
        else:
            print(f"Provided file {filename} is not a json file. Skipping...")


def clean_df(dataframe: pd.DataFrame, column_name: str, phraze: str):
    print(f"\nRemoving rows which do not contain the phraze: '{phraze}'")
    start_time = time.time()
    assert isinstance(dataframe, pd.DataFrame), f"[clean_df] DataFrame expected, got {type(dataframe)} instead"

    mask = dataframe[column_name].astype(str).str.contains(phraze, regex=True, na=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"DataFrame cleaned in time: {elapsed_time:.2f} seconds")
    return dataframe[mask].reset_index(drop=True)



# Generator for feeding word2vec model
class TxtSubdirsCorpus(object):
    def __init__(self, pd_path):
        print("\nCorpus active! Path:", pd_path)
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
