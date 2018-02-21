# import modules & set up logging

import logging, os, zipfile, codecs
from gensim.models import Word2Vec



class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding="utf8"):
                yield line.split()

class ZippedSentences(object):
    def __init__(self, zippedname, number_to_extract):
        self.zippedname = zippedname
        self.number_to_extract = number_to_extract


    def printinfo(self):
        i = 0
        with zipfile.ZipFile(self.zippedname) as myzip:
            for file in myzip.filelist:
                if i <= self.number_to_extract:
                    i += 1
                    with myzip.open(file) as myfile:
                        #myfile = myfile.decode("utf-8")
                        for line in myfile:
                            line = codecs.decode(line, "utf-8")
                            print(line)

    def __iter__(self):
        i = 0
        with zipfile.ZipFile(self.zippedname) as myzip:
            for file in myzip.filelist:
                if i <= self.number_to_extract:
                    i += 1
                    with myzip.open(file) as myfile:
                        for line in myfile:
                            line = codecs.decode(line, "utf-8")
                            yield line.split()

class CBOW:

    def __init__(self, articles_to_learn):
        self.articles_to_learn = articles_to_learn

    def get_model(self):
        zips = ZippedSentences('wiki_flat.zip', self.articles_to_learn) #Extract x number of articles from training set.
        CBOW_model = Word2Vec(sentences=zips, #Sentences to train from
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

class Skip_Gram:

    def __init__(self, articles_to_learn):
        self.articles_to_learn = articles_to_learn

    def get_model(self):
        zips = ZippedSentences('wiki_flat.zip', self.articles_to_learn) #Extract x number of articles from training set.
        Skip_Gram_model = Word2Vec(sentences=zips, #Sentences to train from
                              sg=0, #1 for CBOW, 0 for Skip-gram
                              hs=1, #1 for hierarchical softmax and 0 and non-zero in negative argument then negative sampling is used.
                              negative=1, #0 for no negative sampling and above specifies how many noise words should be drawn. (Usually 5-20 is good).
                              iter=10, #number of epochs.
                              size=100, #feature vector dimensionality
                              min_count=5, #minimum frequency of words required
                              max_vocab_size=None, #How much RAM is allowed, 10 million words needs approx 1GB RAM. None = infinite RAM
                              workers=3, #How many threads are started for training.

                              )
        return Skip_Gram_model


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__))


cbow = CBOW.get_model(CBOW, 10000)
skipgram = Skip_Gram.get_model(Skip_Gram, 10000)

cbow.accuracy(dir_path + '/TestingSet/questions-words.txt')
skipgram.accuracy(dir_path + '/TestingSet/questions-words.txt')

