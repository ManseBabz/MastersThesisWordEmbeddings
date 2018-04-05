import LeaningAlgoImpl.Clustering_Evaluation as cluster
import logging, os

clusterer = cluster.Clustering_Evaluation()

words, correct_values, number_of_clusters, correct_assigns = clusterer.get_words_to_cluster('noun_verb_adj.csv')

clusterer.accuracy([['House', 'Mice', 'Bed', 'Talking', 'Store'], ['Stand', 'Wait', 'Drink'], ['Big', 'Large', 'Beautiful']], correct_assigns, correct_values)