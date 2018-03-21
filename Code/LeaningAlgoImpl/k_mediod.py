
from gensim.models import KeyedVectors

class k_mediod:

    def __init__(self, path_to_test, finished_model):
        self.path = path_to_test
        self.model = finished_model

    def find_clusters(self, k):
        wordses = ["hello", "what", "are", "you", "doing"]
        return k_mediod(k, wordses)

    def k_mediod(self, k, text_to_cluster):

        mediods = []
        clusters = []
        for i in range(0, k):
            mediods.append(text_to_cluster[i])
            cluster = []
            cluster.append(text_to_cluster[i])
            clusters.append(cluster)

        clusters = assign_mediods(mediods, clusters, text_to_cluster)

        previous_cost = total_cost(clusters, mediods)

        new_cost = 0
        while new_cost < previous_cost:
            clusters, mediods, new_cost = bla(clusters, mediods)
        return clusters, mediods

    def bla(self, clusters, mediods):

        old_clusters = clusters
        old_mediods = mediods
        old_cost = total_cost(old_clusters, old_mediods)

        for i in range(0, len(mediods)):
            for word in clusters[i]:
                if word != mediods[i]:
                    mediods[i] = word
                    clusters = reassign_clusters(mediods, clusters)
                    cost = total_cost(clusters, mediods)
                    if cost < old_cost:
                        old_clusters = clusters
                        old_mediods = mediods
                        old_cost = cost

        return old_clusters, old_mediods, old_cost


    def reassign_clusters(self, mediods, clusters):
        new_clusters = []
        for mediod in mediods:
            cluster = []
            cluster.append(mediod)
            new_clusters.append(cluster)

        words = []
        for cluster in clusters:
            for word in cluster:
                words.append(word)
        new_clusters = assign_mediods(mediods, new_clusters, words)
        return new_clusters


    def assign_mediods(self, mediods, clusters, text_to_cluster):
        for word in text_to_cluster:
            closets_mediod = ""
            dist_to_closets = 1
            for mediod in mediods:
                dist_to_mediod = abs(self.model.distance(mediod, word))
                if dist_to_closets > dist_to_mediod:
                    closets_mediod = mediod
                    dist_to_closets = dist_to_mediod
            for i in range(0, len(mediods)):
                if mediods[i] == closets_mediod:
                    clusters[i].append(word)

        return clusters


    def total_cost(self, clusters, mediods):
        total_cost = 0
        for i in range(0, len(mediods)):
            total_cost += cost(clusters[i], mediods[i])
        return total_cost

    def cost(self, cluster, mediod):
        cluster_cost = 0
        for word in cluster:
            cluster_cost += abs(self.model.distance(mediod, word))
        return cluster_cost
