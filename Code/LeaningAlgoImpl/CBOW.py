import logging, os
from LeaningAlgoImpl.Sentence import MySentences
from gensim.models import Word2Vec


class CBOW:

    def get_model(self):
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        sentences1 = MySentences(dir_path + '/DataSet')      # Gets all files from folder at location.
        CBOW_model = Word2Vec(sentences=sentences1, #Sentences to train from
                              sg=1, #1 for CBOW, 0 for Skip-gram
                              hs=1, #1 for hierarchical softmax and 0 and non-zero in negative argument then negative sampling is used.
                              negative=5, #0 for no negative sampling and above specifies how many noise words should be drawn. (Usually 5-20 is good).
                              cbow_mean=0, #0 for sum of context vectors, 1 for mean of context vectors. Only used on CBOW.
                              iter=10, #number of epochs.
                              size=100, #feature vector dimensionality
                              min_count=5, #minimum frequency of words required
                              max_vocab_size=None, #How much RAM is allowed, 10 million words needs approx 1GB RAM. None = infinite RAM
                              workers=3, #How many threads are started for training.

                              )
        return CBOW_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__))

cbow = CBOW.get_model(CBOW)

cbow.accuracy(dir_path + '/TestingSet/questions-words.txt')