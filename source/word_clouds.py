# basic
import os
from pathlib import Path
import pandas as pd
import numpy as np

# case-specific
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plot_wordcloud(text: str, title: str = " "):
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=24, fontweight='bold', color='black', loc='center')
    plt.show()


def divide_and_plot(df: pd.DataFrame, cluster: str):
    number_of_clusters = df[cluster].nunique()
    for i in range(number_of_clusters):
        grouped = df.groupby(df[cluster])
        df_to_print = grouped.get_group(i)
        cloud = ''
        for t in df_to_print['clean_text']:
            cloud += str(t).replace("'", '') + ' '
        plot_wordcloud(cloud, title=f"Wordcloud of {i}")

