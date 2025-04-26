# basic
import pandas as pd

# case-specific
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.decomposition import PCA


def plot_dimentions(data: pd.DataFrame, rdims: int = 2):
    try:
        isinstance(data, list)
        if rdims > 2:
            pca = PCA(n_components=2)
            x_principal = pca.fit_transform(data)
            x_principal = pd.DataFrame(x_principal, columns=['Dim 1', 'Dim 2'])
            plt.title(f"Distance of document vectors (PCA)", fontsize=12, fontweight="bold")

        elif rdims == 2:
            x_principal = data
            plt.figure(figsize=(7, 5))
            sns.set_theme(style="ticks")
            sns.relplot(data=x_principal,
                        x=x_principal['Dim 1'],
                        y=x_principal['Dim 2'],
                        s=25,
                        alpha=0.9,
                        edgecolors='black',
                        legend="auto")
            plt.title(f"Distance of document vectors", fontsize=12, fontweight="bold")
        else:
            print(f"You declared wrong number of dimentions {rdims}")
            return

        plt.xlabel("Dimention no.1", fontsize=12, fontweight="bold")
        plt.ylabel("Dimention no.2", fontsize=12, fontweight="bold")
        max_len = max(x_principal['Dim 1'].max(), x_principal['Dim 2'].max())
        min_len = min(x_principal['Dim 1'].min(), x_principal['Dim 2'].min())
        plt.xlim(min_len - 4, max_len + 4)
        plt.ylim(min_len - 4, max_len + 4)
        plt.tight_layout()
        plt.show()
    except TypeError:
        print(f"Data is not a list!")


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

