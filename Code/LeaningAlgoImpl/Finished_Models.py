import logging, os
from LeaningAlgoImpl.ToolsPackage.Sentence import MySentences
from LeaningAlgoImpl.ToolsPackage.UnZipper import ZippedSentences
from LeaningAlgoImpl import k_mediod
from gensim.models import KeyedVectors

class Finished_Models:

    def get_model(self, path):
        model = KeyedVectors.load(path)
        self.model = model

    def acc(self, testset = 'questions-words.txt'):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        print(dir_path + '/TestingSet/' + testset)
        return self.model.accuracy(dir_path + '/TestingSet/' + testset)

    def most_similar(self, positive_words, negative_words):
        self.model.most_similar(positive=positive_words, negative=negative_words) # Like positive=['woman', 'king'], negative=['man']

    def doesnt_match(self, sentence):
        self.model.doesnt_match(sentence.split()) # "breakfast cereal dinner lunch" -> 'cereal'

    def similarity(self, word1, word2):
        return self.model.similarity(word1, word2) # 'woman', 'man' -> 0.73723527

    def human_similarity_test(self, testset = 'wordsim353.tsv'):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        return self.model.evaluate_word_pairs(dir_path + '/TestingSet/' + testset)

    def clustering_test(self):
        mediod = k_mediod.k_mediod("", self.model)
        clusters, mediods = mediod.find_clusters(2)
        print(clusters)
        print(mediods)

    def get_vocabulary(self):
        print(self.model.wv.vocab)

    def __init__(self):
        self.model = None