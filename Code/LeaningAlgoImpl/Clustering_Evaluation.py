from LeaningAlgoImpl import k_mediod
import csv, os

class Clustering_Evaluation:

    def find_clusters_with_k_mediod(self, words, model, k):
        clusterer = k_mediod(words, model)
        clusters = clusterer.find_clusters(k)
        return clusters


    def get_words_to_cluster(self, filename):
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/Evaluation/Clustering/'
        words = []
        clustering_values = []
        correct_assigns = []
        with open(dir_path + filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                words.append(row[0])
                clustering_values.append(row[1])
                correct_assigns.append(row)
        clustering_values = set(clustering_values)
        number_of_clusters = len(clustering_values)
        return words, clustering_values, number_of_clusters, correct_assigns


    def accuracy(self, clusters, correct_assigns, cluster_values):
        assigned_clusters = []
        for cluster in clusters:
            assigned_cluster = []
            for word in cluster:
                for wordpair in correct_assigns:
                    if word == wordpair[0]:
                        assigned_cluster.append(wordpair)
            assigned_clusters.append(assigned_cluster)
        print('TESTING')
        print(assigned_clusters)

        purity_of_clusters = []
        for assigned_cluster in assigned_clusters:
            purity_of_cluster = []
            for cluster_value in cluster_values:
                number_of_wordtype = 0
                for wordpair in assigned_cluster:
                    if wordpair[1] == cluster_value:
                        number_of_wordtype += 1
                percentage_of_wordtype = number_of_wordtype / len(assigned_cluster)
                purity_of_cluster.append([cluster_value, percentage_of_wordtype])

            purity_of_clusters.append(purity_of_cluster)

        #print(purity_of_clusters)
        return purity_of_clusters

    def __init__(self):
        print('Initialized')