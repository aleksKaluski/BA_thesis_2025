## Clustering pipeline with text embedings
This repository contains a complete pipeline for processing, embedding, and clustering scientific texts, originally extraced from peS2o using Lingo4G and stored as JSON files.

**Pipeline Overview**:
1) Split the files into paragraphs based on line break characters.
  * tag the text using spaCy.
  * discard lines that are too short.
  * store the processed data in a pandas DataFrame.

2) Create a Corpus class object, which acts as a generator and is used to stream texts into Word2Vec incrementally.

3) Train several Word2Vec models and evaluate them using Google's analogy dataset as well as selected word pairs.

  * Using the best-performing model, we assign a vector to each paragraph.

4) Reduce vector dimensions using UMAP and visualize document distances (for visualization, we always apply PCA to reduce to 2 dimensions).

  * UMAP reduction is performed to 4 dimensions. We then work on a DataFrame representing the full corpus, created via the generator.

5) Cluster the vectors using several methods:

  * Mini-batch K-Means clustering — computationally efficient, used to estimate the optimal number of clusters.

  * Gaussian Mixture Models — clustering based on probabilistic Gaussian distributions.

  * Hierarchical Clustering — additional hierarchical clustering.

  * HDBSCAN — density-based clustering.

6) Visualize the contents of individual clusters using word clouds.

DISCLAIMER: not all the sources of code are inlncluded here (work in progress)

**Sources**
* https://github.com/JonasTriki/masters-thesis-ml/tree/master
* https://programminghistorian.org/en/lessons/clustering-visualizing-word-embeddings
* https://programminghistorian.org/en/lessons/clustering-with-scikit-learn-in-python
