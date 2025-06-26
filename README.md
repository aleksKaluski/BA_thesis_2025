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


**Sources**
* https://github.com/JonasTriki/masters-thesis-ml/tree/master
* https://programminghistorian.org/en/lessons/clustering-visualizing-word-embeddings
* https://programminghistorian.org/en/lessons/clustering-with-scikit-learn-in-python
* [Word Embedding using Word2Vec - GeeksforGeeks](https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/)
* [lingo_tunning](https://get.carrotsearch.com/lingo4g/latest-v1/doc/#introduction)
* [lingo_tunning_labels](https://get.carrotsearch.com/lingo4g/latest-v1/doc/#labels)
* [Lingo4G, clustering engine reference](https://get.carrotsearch.com/lingo4g/latest-v1/doc/#labels.arrangement.algorithm.inputPreference)
* [lingo_exp](https://get.carrotsearch.com/lingo4g/latest-v1/doc/#explorer-experiments)
* [Clustering and Visualising Documents using Word Embeddings | Programming Historian](https://programminghistorian.org/en/lessons/clustering-visualizing-word-embeddings)
* [Clustering with Scikit-Learn in Python | Programming Historian](https://programminghistorian.org/en/lessons/clustering-with-scikit-learn-in-python)
* [Clustering and Visualising Documents using Word Embeddings | Programming Historian](https://programminghistorian.org/en/lessons/clustering-visualizing-word-embeddings#prerequisites)
* [read_json pandas - Szukaj w Google](https://www.google.com/search?q=read_json+pandas&rlz=1C1CHZN_plPL952PL952&oq=read_json+pandas&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIJCAEQABgTGIAEMgoIAhAAGBMYFhgeMgoIAxAAGBMYFhgeMgoIBBAAGBMYFhgeMgoIBRAAGBMYFhgeMgwIBhAAGAoYExgWGB4yCggHEAAYExgWGB4yCggIEAAYExgWGB4yCggJEAAYExgWGB7SAQg0MTI0ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
* [pandas.read_json — pandas 2.2.3 documentation](https://pandas.pydata.org/docs/reference/api/pandas.read_json.html)
* [Understanding and Creating Word Embeddings | Programming Historian](https://programminghistorian.org/en/lessons/understanding-creating-word-embeddings)
* [Corpus Analysis with spaCy | Programming Historian](https://programminghistorian.org/en/lessons/corpus-analysis-with-spacy)
* [The Illustrated Word2vec – Jay Alammar – Visualizing machine learning one concept at a time.](https://jalammar.github.io/illustrated-word2vec/)
* [WWP](https://www.wwp.northeastern.edu/)
* [Understanding and Using Common Similarity Measures for Text Analysis | Programming Historian](https://programminghistorian.org/en/lessons/common-similarity-measures)
* [Principal Component Analysis (PCA) Explained | Built In](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)
* [Principal component analysis: a review and recent developments | Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences](https://royalsocietypublishing.org/doi/10.1098/rsta.2015.0202)
* [How UMAP Works — umap 0.5.8 documentation](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html)
* [[1802.03426] UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426)
* [operating system - How to move to one folder back in python - Stack Overflow](https://stackoverflow.com/questions/12280143/how-to-move-to-one-folder-back-in-python)
* [Hierarchical clustering (scipy.cluster.hierarchy) — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
* [Word Embeddings as Metric Recovery in Semantic Spaces - ACL Anthology](https://aclanthology.org/Q16-1020/)
* [Processing large JSON files in Python without running out of memory](https://pythonspeed.com/articles/json-memory-streaming/)
* [Data streaming in Python: generators, iterators, iterables | RARE Technologies](https://rare-technologies.com/data-streaming-in-python-generators-iterators-iterables/)
* [Articles: Speed up your data science and scientific computing code](https://pythonspeed.com/datascience/#memory)
* [Singular Value Decomposition (SVD) - GeeksforGeeks](https://www.geeksforgeeks.org/singular-value-decomposition-svd/)
* [Tutorial 1: Comparing Word Embedding Models - YouTube](https://www.youtube.com/watch?v=WbzPZZKJRJA)
* [[1411.2738] word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738)
* [Full article: Digital begriffsgeschichte: Tracing semantic change using word embeddings](https://www.tandfonline.com/doi/full/10.1080/01615440.2020.1760157#abstract)
* [Artificial Intelligence Business Concept In Retro Collage Style With Human Hand And Artificial Hand Touching Above World Map Stock Illustration - Download Image Now - iStock](https://www.istockphoto.com/en/vector/artificial-intelligence-business-concept-in-retro-collage-style-with-human-hand-and-gm2160633336-581156071)
* [linkage — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
* [python - How can I iterate over files in a given directory? - Stack Overflow](https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory)
* [masters-thesis-ml/code/analysis_of_word_embeddings/clustering_word_categories/country_continent_cluster_analysis.ipynb at master · JonasTriki/masters-thesis-ml](https://github.com/JonasTriki/masters-thesis-ml/blob/master/code/analysis_of_word_embeddings/clustering_word_categories/country_continent_cluster_analysis.ipynb)
* [masters-thesis-ml/code/general_machine_learning_methods/thesis_background_plots.ipynb at master · JonasTriki/masters-thesis-ml](https://github.com/JonasTriki/masters-thesis-ml/blob/master/code/general_machine_learning_methods/thesis_background_plots.ipynb)
* [Hierarchical clustering: structured vs unstructured ward — scikit-learn 1.6.1 documentation](https://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html)
* [pandas - How to get Agglomerative Clustering "Centroid" in python Scikit-learn - Stack Overflow](https://stackoverflow.com/questions/56456572/how-to-get-agglomerative-clustering-centroid-in-python-scikit-learn)
* [AgglomerativeClustering with Cluster Centers | by Tushar Sinha | Medium](https://tushar-osc.medium.com/agglomerativeclustering-with-cluster-centers-e5d409c724d1)
* [(50) Matplotlib Legend Tutorial || matplotlib legend outside of graph || Matplotlib Tips - YouTube](https://www.youtube.com/watch?v=lnfGvdCqGYs)
* [Python Tutorial 4: Tokenization, Lemmatization, and Frequency Lists | Introduction to Corpus Analysis With Python 3](https://kristopherkyle.github.io/corpus-analysis-python/Python_Tutorial_4.html)
* [silhouette_score — scikit-learn 1.6.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
* [2.3. Clustering — scikit-learn 1.6.1 documentation](https://scikit-learn.org/stable/modules/clustering.html#)
* [Comparison of the K-Means and MiniBatchKMeans clustering algorithms — scikit-learn 1.6.1 documentation](https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html)
* [GMM covariances — scikit-learn 1.6.1 documentation](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py)
* [Gaussian Mixture Model Selection — scikit-learn 1.6.1 documentation](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py)
* [Variable Selection for Clustering with Gaussian Mixture Models | Biometrics | Oxford Academic](https://academic.oup.com/biometrics/article-abstract/65/3/701/7331797)
* [Disk full, how to change download folder, downloading datasets to a new folder - Beginners - Hugging Face Forums](https://discuss.huggingface.co/t/disk-full-how-to-change-download-folder-downloading-datasets-to-a-new-folder/136402)
* [models.phrases – Phrase (collocation) detection — gensim](https://radimrehurek.com/gensim/models/phrases.html#module-gensim.models.phrases)
* [gitignore.io - Create Useful .gitignore Files For Your Project](https://www.toptal.com/developers/gitignore/)
* [Gaussian Mixture Model Explained | Built In](https://builtin.com/articles/gaussian-mixture-model)
* [quickstart.ipynb - Colab](https://colab.research.google.com/github/optuna/optuna-examples/blob/main/quickstart.ipynb#scrollTo=I4QdanM31YKg)
* [python - How to optimize for multiple metrics in Optuna - Stack Overflow](https://stackoverflow.com/questions/69071684/how-to-optimize-for-multiple-metrics-in-optuna)
* [python 3.x - How to speed up a spacy pipeline with the nlp.pipe pattern? - Stack Overflow](https://stackoverflow.com/questions/58320922/how-to-speed-up-a-spacy-pipeline-with-the-nlp-pipe-pattern)
* [Accelerate NLP preprocessing pipeline by optimizing a batch-processing loop | by Tristan Vanrullen | Medium](https://tristanv.medium.com/accelerate-nlp-preprocessing-pipeline-by-optimizing-a-batch-processing-loop-cf18dc7ee036)
* [Different ways to iterate over rows in Pandas Dataframe | GeeksforGeeks](https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/)
* [Parameter Selection for HDBSCAN* — hdbscan 0.8.1 documentation](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html)
