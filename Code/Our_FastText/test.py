import Own_Fast_Text as fasttext
import logging, os, zipfile, codecs

class ZippedSentences(object):
    # Total files in wiki_flat -> 4.854.231
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

class Fast_Text:

    def __init__(self, articles_to_learn):
        self.articles_to_learn = articles_to_learn


    def get_model(self):
        zips = ZippedSentences('wiki_flat.zip', self.articles_to_learn) #Extract x number of articles from training set.
        Fast_Text_model = fasttext.FastText(sentences=zips, #Sentences to train from
                              sg=1, #0 for CBOW, 1 for Skip-gram
                              hs=1, #1 for hierarchical softmax and 0 and non-zero in negative argument then negative sampling is used.
                              negative=1, #0 for no negative sampling and above specifies how many noise words should be drawn. (Usually 5-20 is good).
                              iter=2, #number of epochs.
                              size=100, #feature vector dimensionality
                              min_count=5, #minimum frequency of words required
                              max_vocab_size=None, #How much RAM is allowed, 10 million words needs approx 1GB RAM. None = infinite RAM
                              workers=3, #How many threads are started for training.
                              min_n=3, #Minimum length of char n-grams for word representations, (4 means a word of 5 will be split into 4 parts, an extra beginning part and end part is added to words)
                              max_n=6, #Maximum length of char n-grams
                              adaptive=1, #Use adaptive version if 1
                              word_ngrams=1 #1 means using char n-grams, 0 equals word2vec.
                              )
        return Fast_Text_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

faster = Fast_Text(1000)
fast = faster.get_model()




dir_path = os.path.dirname(os.path.realpath(__file__))

fast.accuracy(dir_path + '/TestingSet/questions-words.txt')
