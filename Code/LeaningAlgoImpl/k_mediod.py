
from gensim.models import KeyedVectors
from random import shuffle

class k_mediod:


    def find_clusters(self, testset):
        #words_to_cluster = ["hello", "what", "are", "you", "doing"]

        ok_vocab = self.model.get_upper_vocab()


        oov_free_testset = []
        oov_words = []
        for cluster in testset:
            oov_free_cluster = []
            for word in cluster:
                word = word.upper()
                if word in ok_vocab:
                    oov_free_cluster.append(word)
                else:
                    oov_words.append(word)
            oov_free_testset.append(oov_free_cluster)
        #print(oov_words)

        old_vocab = self.model.get_vocabulary()
        self.model.set_vocabulary(self.model.get_upper_vocab())

        clusters, mediods, cost = k_mediod.k_mediods(self, oov_free_testset, len(oov_free_testset))

        self.model.set_vocabulary(old_vocab)

        return clusters, mediods, cost, oov_words

    def k_mediods(self, text_to_cluster, k):

        all_words = []
        for cluster in text_to_cluster:
            for word in cluster:
                all_words.append(word)
        #print(all_words)

        shuffle(all_words)
        #print(all_words)

        mediods = []
        for i in range(0, k):
            mediods.append(all_words[i])


        #print(mediods)

        clusters = k_mediod.assign_mediods(self, mediods, all_words)

        #print("initial clusters")
        #print(clusters)

        previous_cost = k_mediod.total_cost(self, clusters, mediods)

        #print("initial cost")
        #print(previous_cost)

        clusters, mediods, new_cost = k_mediod.update_clusters(self, clusters, mediods)
        #print(clusters)
        #print(mediods)
        #print(new_cost)
        while new_cost < previous_cost:
            previous_cost = new_cost
            clusters, mediods, new_cost = k_mediod.update_clusters(self, clusters, mediods)
            #print(clusters)
            #print(mediods)
            #print(previous_cost)
            #print(new_cost)
            #print(new_cost < previous_cost)


        return clusters, mediods, new_cost

    def update_clusters(self, clusters, mediods):

        old_clusters = list(clusters)
        old_mediods = list(mediods)
        old_cost = k_mediod.total_cost(self, old_clusters, old_mediods)

        for i in range(0, len(mediods)):
            for cluster in old_clusters:
                for word in cluster:
                    if word not in old_mediods:
                        mediods[i] = word
                        clusters = k_mediod.reassign_clusters(self, clusters, mediods)
                        cost = k_mediod.total_cost(self, clusters, mediods)
                        if cost < old_cost:
                            return clusters, mediods, cost
        return old_clusters, old_mediods, old_cost

    def reassign_clusters(self, clusters, mediods):
        all_words = []
        for cluster in clusters:
            for word in cluster:
                all_words.append(word)
        #print(all_words)
        new_clusters = k_mediod.assign_mediods(self, mediods,  all_words)
        return new_clusters


    def assign_mediods(self, mediods, all_words):
        clusters = []
        for word in mediods:
            clusters.append([word])
        #print(clusters)
        for word in all_words:
            closets_mediod = ""
            dist_to_closets = 1
            for mediod in mediods:
                dist_to_mediod = abs(self.model.distance(mediod, word))
                if dist_to_closets > dist_to_mediod:
                    closets_mediod = mediod
                    dist_to_closets = dist_to_mediod
            for i in range(0, len(mediods)):
                if mediods[i] == closets_mediod and mediods[i] != word:
                    clusters[i].append(word)

        return clusters


    def total_cost(self, clusters, mediods):
        total_cost = 0
        for mediod in mediods:
            for cluster in clusters:
                if mediod in cluster:
                    total_cost += k_mediod.cost(self, cluster, mediod)

        return total_cost

    def cost(self, cluster, mediod):
        cluster_cost = 0
        for word in cluster:
            cluster_cost += abs(self.model.distance(word, mediod))
        return cluster_cost

    def __init__(self, finished_model):
        self.model = finished_model