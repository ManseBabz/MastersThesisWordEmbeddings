import LeaningAlgoImpl.Clustering_Evaluation as cluster
import logging, os

class clustering_test:

    def get_accuracy(self, filename, model):
        clusterer = cluster.Clustering_Evaluation()
        words, correct_values, number_of_clusters, correct_assigns = clusterer.get_words_to_cluster(filename)
        clusters = clusterer.find_clusters_with_k_mediod(self, words, model, number_of_clusters)

        cluster_purity = clusterer.accuracy(clusters, correct_assigns, correct_values)

        print(cluster_purity)

    def __init__(self):
        print('Starting clustering test ready for get_accuracy with clustering test file and model')