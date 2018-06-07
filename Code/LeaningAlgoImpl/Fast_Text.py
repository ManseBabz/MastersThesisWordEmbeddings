import logging, os
from LeaningAlgoImpl.ToolsPackage.Sentence import MySentences
from LeaningAlgoImpl.ToolsPackage.UnZipper import ZippedSentences

from LeaningAlgoImpl import k_mediod

from gensim.models import KeyedVectors
from gensim.models import fasttext


class Fast_Text:
    def get_model(self, language, hs =1, negative= 5, cbow_mean=0, iter= 10, size=100, min_count=5, max_vocab_size=1000000, workers=3, articles_to_learn=1000, randomTrain=False):
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        if (self.dev_mode):
            sentences1 = MySentences(dir_path + '/DataSet')  # Gets all files from folder at location.
        else:
            print("Training model, be aware this is on a real trainingset, so it might take a while")
            if (language == 'English'):
                sentences1 = ZippedSentences(dir_path + '/RealDataSet/wiki_flat.zip', articles_to_learn,
                                             randomTrain)  # Make train-data from a large sample of data using articles_to_learn articles
            elif (language == 'Danish'):
                sentences1 = ZippedSentences(dir_path + '/RealDataSet/' + language + '/danishTrainingSet.zip',
                                             articles_to_learn, randomTrain)
            else:
                raise ValueError('No apropiate trainings set')
        Fast_Text_model = fasttext.FastText(sentences=sentences1,  # Sentences to train from
                                            sg=1,  # 0 for CBOW, 1 for Skip-gram
                                            hs=hs,# 1 for hierarchical softmax and 0 and non-zero in negative argument then negative sampling is used.
                                            negative=negative,# 0 for no negative sampling and above specifies how many noise words should be drawn. (Usually 5-20 is good).
                                            cbow_mean=cbow_mean,# 0 for sum of context vectors, 1 for mean of context vectors. Only used on CBOW.
                                            iter=iter,  # number of epochs.
                                            size=size,  # feature vector dimensionality
                                            min_count=min_count,  # minimum frequency of words required
                                            max_vocab_size=max_vocab_size,# How much RAM is allowed, 10 million words needs approx 1GB RAM. None = infinite RAM
                                            workers=workers,  # How many threads are started for training.
                                            )
        self.model=Fast_Text_model
        return Fast_Text_model

    def acc(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.model.accuracy(dir_path + '/TestingSet/questions-words.txt')

    def predict(self, positive_word_list, negative_word_list):
        try:
            return self.finished_model.most_similar(positive=positive_word_list, negative=negative_word_list)
        except KeyError:
            return []

    def similarity(self, word_one, word_two):
        try:
            return self.finished_model.similarity(word_one, word_two)
        except KeyError:
            return None

    def load_model(self, name, language):
        print("Great you were able to load a model, no need to create a new one")
        if (self.dev_mode):
            dir_path = os.path.dirname(os.path.realpath(__file__)) + "/DevModels/"+language+'/' + name
            self.finished_model = KeyedVectors.load(dir_path)
        else:
            dir_path = os.path.dirname(os.path.realpath(__file__)) + "/Models/" +language+'/'+ name
            self.finished_model = KeyedVectors.load(dir_path)

    def save_model(self, name, language):
        if(self.dev_mode):
            print(self.model)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.model.save(dir_path + "/DevModels/"+language+'/' + name)
        else:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.model.save(dir_path + "/Models/"+language+'/' + name)

    def finished_training(self):
        self.finished_model = self.model.wv

    def save_finished_model(self, name, language):
        if (self.dev_mode):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.finished_model.save(dir_path + "/DevModels/"+language+'/' + name)
        else:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.finished_model.save(dir_path + "/Models/"+language+'/' + name)

    def clustering_test(self):
        mediod = k_mediod.k_mediod("", self.finished_model)
        clusters, mediods = mediod.find_clusters(2)
        print(clusters)
        print(mediods)

    def __init__(self, dev_mode=False):
        self.model = None
        self.finished_model =None
        self.dev_mode = dev_mode