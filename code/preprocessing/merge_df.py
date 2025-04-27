# basic
import os
from pathlib import Path
import pandas as pd
import numpy as np

# case-specific
from tqdm import tqdm


def iter_large_df(top_dir):
    print(f"\nMerging dataframes. Iterating documents in {top_dir}...")
    for root, _, files in os.walk(top_dir):
        for file in files:
            if file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                df = pd.read_pickle(file_path)
                for _, row in df.iterrows():
                    yield row


def merge_df(dir_dfs: str) -> pd.DataFrame:
    path = Path(dir_dfs)
    assert path.exists(), f"Provided path {dir_dfs} does not exist"
    assert path.is_dir(), f"Provided path {dir_dfs} is not a directory!"

    gen = iter_large_df(dir_dfs)

    d = []
    row_counter = 0

    for r in gen:
        row_counter += 1
        if row_counter % 10000 == 0:
            print(f"Processed {row_counter} rows...")

        d.append([
            r["clean_text"],
            r["text"],
            r["semantic_id"],
            r["date"],
            r["group"]
        ])

    df_full = pd.DataFrame(d, columns=["clean_text", "text", "semantic_id", "date", "group"])
    print(f"Done! Total rows merged: {row_counter}")
    return df_full
