# basic
import os
from pathlib import Path
import pandas as pd
import numpy as np

# case-specific
import umap

# count the mean vector of a doc
def document_vector(word2vec_model, doc_tokens):
    vectors = []
    for w in doc_tokens:
        if w in word2vec_model.wv:
            # take the vector of w
            vectors.append(word2vec_model.wv[w])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)


def add_document_vector(df: pd.DataFrame, model) -> pd.DataFrame:
    df["vector"] = df['clean_text'].apply(lambda tokens: document_vector(model, tokens))
    return df


# Extract the embedding from a list-type column and make it a df of it's own
def x_from_df(df: pd.DataFrame, col: str = 'Embedding') -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        cols = ['E' + str(x) for x in np.arange(0, len(df[col].iloc[0]))]
        return pd.DataFrame(df[col].tolist(), columns=cols, index=df.index)

    elif isinstance(df, str):
        my_file = Path(df)
        if my_file.is_file():
            try:
                df = pd.read_pickle(df)
                cols = ['E' + str(x) for x in np.arange(0, len(df[col].iloc[0]))]
                return pd.DataFrame(df[col].tolist(), columns=cols, index=df.index)

            except TypeError:
                print(f"You passed a string {df} as a df, but it's not a pickle file.")
        else:
            print(f"You passed a string {df} as a df, but it's neither a file, nor a dataframe.")



def reduce_dimentionality(df_vectors: pd.DataFrame,
                          df_normal: pd.DataFrame,
                          seed: int = 43,
                          dmeasure: str = 'euclidean',
                          rdims: int = 4,
                          min_dist: float = 0.01,
                          neighbors: int = 25) -> pd.DataFrame:

    print(f"UMAP dimensionality reduction to {rdims} dimensions with '{dmeasure}' distance measure.")

    # random seed
    rs = seed
    reducer = umap.UMAP(
        n_neighbors=neighbors,
        min_dist=min_dist,
        n_components=rdims,
        random_state=rs)

    df_normal = df_normal.reset_index(drop=True)
    df_vectors = df_vectors.reset_index(drop=True)
    # matrix with our vectors
    embeding_matrix = reducer.fit_transform(df_vectors)

    # comvert embeding_matrix to a df,
    # then merge it with basic df
    embedded_dict = {}
    for i in range(0, embeding_matrix.shape[1]):
        embedded_dict[f"Dim {i + 1}"] = embeding_matrix[:, i]

    dfe = pd.DataFrame(embedded_dict, index=df_normal.index)
    del (embedded_dict)
    projected = df_normal.join(dfe)

    return projected