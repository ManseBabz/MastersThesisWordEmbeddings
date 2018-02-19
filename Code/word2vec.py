# import modules & set up logging

import logging, os
from gensim.models import Word2Vec


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__))


sentences = MySentences(dir_path + '/DataSet')

model = Word2Vec(sentences, min_count=1)

model.accuracy(dir_path + '/TestingSet/questions-words.txt')