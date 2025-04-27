import os
import pandas as pd
from code.preprocessing import load_data as ld
import spacy

os.chdir(r"C:/BA_thesis/BA_v2_31.03")
print(f"working directory: {os.getcwd()}")

input_path = os.getcwd() + '/files/mock'

df = pd.read_pickle('files/dfs/dp_analysis_10k_0_prp.pkl')
print(df.head())
print(df.info())