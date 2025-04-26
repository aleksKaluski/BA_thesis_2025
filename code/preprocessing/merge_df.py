# basic
import os
from pathlib import Path
import pandas as pd
import numpy as np

# case-specific
from tqdm import tqdm


def iter_large_df(top_dir):
    print(f"Mearging dataframe ha started. Iterating documents in {top_dir}...")
    for root, _, files in os.walk(top_dir):
        for file in files:
            if file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                df = pd.read_pickle(file_path)
                for _, row in df.iterrows():
                    yield row


def merge_df(dir_dfs: str) -> pd.DataFrame:
    path = Path(dir_dfs)
    assert os.path.exists(path), f"Provided path {dir_dfs} does not exist"
    assert os.path.isdir(path), f"Provided path {dir_dfs} is not dir path!"

    gen = iter_large_df(dir_dfs)

    # temporary date container
    d = []
    for r in gen:
        d.append([
            r["clean_text"],
            r["semantic_id"],
            r["date"],
            r["group"]
        ])

    df_full = pd.DataFrame(d, columns=["clean_text", "semantic_id", "date", "group"])
    print("Dfs merged sucesfully!")
    print('*'*50)
    return df_full
