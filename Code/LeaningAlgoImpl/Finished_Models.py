import logging, os
from LeaningAlgoImpl.ToolsPackage.Sentence import MySentences
from LeaningAlgoImpl.ToolsPackage.UnZipper import ZippedSentences
from gensim.models import KeyedVectors

class Finished_Models:

    def get_model(self, path):
        model = KeyedVectors.load(path)
        self.model = model

    def acc(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.model.accuracy(dir_path + '/TestingSet/questions-words.txt')

    def most_similar(self, positive_words, negative_words):
        self.model.most_similar(positive=positive_words, negative=negative_words) # Like positive=['woman', 'king'], negative=['man']

    def doesnt_match(self, sentence):
        self.model.doesnt_match(sentence.split()) # "breakfast cereal dinner lunch" -> 'cereal'

    def similarity(self, word1, word2):
        self.model.similarity(word1, word2) # 'woman', 'man' -> 0.73723527

    def __init__(self):
        self.model = None