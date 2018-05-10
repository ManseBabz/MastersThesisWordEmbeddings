import logging, os
from LeaningAlgoImpl.ToolsPackage.Sentence import MySentences
from LeaningAlgoImpl.ToolsPackage.UnZipper import ZippedSentences
from LeaningAlgoImpl import k_mediod
from LeaningAlgoImpl import danish_keyedvectors
from gensim.models import KeyedVectors

from gensim import utils, matutils

logger = logging.getLogger(__name__)

class Finished_Models:

    def get_model(self, path):
        model = KeyedVectors.load(path)
        self.model = model

    def acc(self, testset = 'questions-words.txt'):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        print(dir_path + '/TestingSet/' + testset)
        return self.model.accuracy(dir_path + '/TestingSet/' + testset)

    def danish_acc(self, testset = 'danish-topology.txt'):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        print(dir_path + '/TestingSet/' + testset)
        return self.own_accuracy(dir_path + '/TestingSet/' + testset)

    def special_danish_acc(self, testset = 'special-danish-topology.txt'):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        print(dir_path + '/TestingSet/' + testset)
        return self.special_danish_accuracy(dir_path + '/TestingSet/' + testset)

    def get_acc_results(self, testset = 'danish-topology.txt'):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        print(dir_path + '/TestingSet/' + testset)
        return self.enseemble_results(dir_path + '/TestingSet/' + testset)

    def most_similar(self, positive_words, negative_words):
        return self.model.most_similar(positive=positive_words, negative=negative_words, topn = 1) # Like positive=['woman', 'king'], negative=['man']

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
        return self.model.wv.vocab

    def set_vocabulary(self, vocab):
        self.model.wv.vocab = vocab

    def __init__(self):
        self.model = None

    @staticmethod
    def log_accuracy(section):
        correct, incorrect = len(section['correct']), len(section['incorrect'])
        if correct + incorrect > 0:
            logger.info(
                "%s: %.1f%% (%i/%i)",
                section['section'], 100.0 * correct / (correct + incorrect), correct, correct + incorrect
            )

    def own_accuracy(self, questions):
        """
        Compute accuracy of the model. `questions` is a filename where lines are
        4-tuples of words, split into sections by ": SECTION NAME" lines.
        See questions-words.txt in
        https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip
        for an example.

        The accuracy is reported (=printed to log and returned as a list) for each
        section separately, plus there's one aggregate summary at the end.

        Use `restrict_vocab` to ignore all questions containing a word not in the first `restrict_vocab`
        words (default 30,000). This may be meaningful if you've sorted the vocabulary by descending frequency.
        In case `case_insensitive` is True, the first `restrict_vocab` words are taken first, and then
        case normalization is performed.

        Use `case_insensitive` to convert all words in questions and vocab to their uppercase form before
        evaluating the accuracy (default True). Useful in case of case-mismatch between training tokens
        and question words. In case of multiple case variants of a single word, the vector for the first
        occurrence (also the most frequent if vocabulary is sorted) is taken.

        This method corresponds to the `compute-accuracy` script of the original C word2vec.

        """
        ok_vocab = self.get_vocabulary()
        print("ok vocab")
        #print(ok_vocab)
        new_vocab = [(w, self.model.wv.vocab[w]) for w in ok_vocab]
        print("not dict")
        #new_vocab = [w.upper() for w in ok_vocab]
        #print(new_vocab)
        new_vocab = {w.upper(): v for w, v in new_vocab}
        new_vocab = dict(new_vocab)
        #print(new_vocab)




        sections, section = [], None
        wrong_predictions = []
        for line_no, line in enumerate(utils.smart_open(questions)):
            # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
            line = utils.to_unicode(line)
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    self.log_accuracy(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
                try:
                    a, b, c, expected = [word.upper() for word in line.split()]
                except ValueError:
                    logger.info("skipping invalid line #%i in %s", line_no, questions)
                    continue
                if a not in new_vocab or b not in new_vocab or c not in new_vocab or expected not in new_vocab:
                    if a not in new_vocab:
                        print("Dont know: " + a)
                    if b not in new_vocab:
                        print("Dont know: " + b)
                    if c not in new_vocab:
                        print("Dont know: " + c)
                    if expected not in new_vocab:
                        print("Dont know: " + expected)
                    logger.debug("skipping line #%i with OOV words: %s", line_no, line.strip())
                    continue

                original_vocab = self.get_vocabulary()
                self.set_vocabulary(new_vocab)
                ignore = {a, b, c}  # input words to be ignored

                # find the most likely prediction, ignoring OOV words and input words
                sims = self.most_similar(positive_words=[b, c], negative_words=[a])
                #print("sims")
                #print(sims)
                self.set_vocabulary(original_vocab)

                predicted = sims[0][0]
                predicted = predicted.upper()
                #print(predicted)
                if predicted == expected:
                    section['correct'].append((a, b, c, expected))
                else:
                    wrong_message = a + " " + b + " " + c + ", predicted: " + predicted + ", should have been: " + expected
                    section['incorrect'].append((a, b, c, expected))
                    wrong_predictions.append(wrong_message)
        if section:
            # store the last section, too
            sections.append(section)
            self.log_accuracy(section)

        total = {
            'section': 'total',
            'correct': sum((s['correct'] for s in sections), []),
            'incorrect': sum((s['incorrect'] for s in sections), []),
        }
        self.log_accuracy(total)
        sections.append(total)
        print(wrong_predictions)
        return sections


    def special_danish_accuracy(self, questions):
        """
        Compute accuracy of the model. `questions` is a filename where lines are
        4-tuples of words, split into sections by ": SECTION NAME" lines.
        See questions-words.txt in
        https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip
        for an example.

        The accuracy is reported (=printed to log and returned as a list) for each
        section separately, plus there's one aggregate summary at the end.

        Use `restrict_vocab` to ignore all questions containing a word not in the first `restrict_vocab`
        words (default 30,000). This may be meaningful if you've sorted the vocabulary by descending frequency.
        In case `case_insensitive` is True, the first `restrict_vocab` words are taken first, and then
        case normalization is performed.

        Use `case_insensitive` to convert all words in questions and vocab to their uppercase form before
        evaluating the accuracy (default True). Useful in case of case-mismatch between training tokens
        and question words. In case of multiple case variants of a single word, the vector for the first
        occurrence (also the most frequent if vocabulary is sorted) is taken.

        This method corresponds to the `compute-accuracy` script of the original C word2vec.

        """
        ok_vocab = self.get_vocabulary()
        print("ok vocab")
        #print(ok_vocab)
        new_vocab = [(w, self.model.wv.vocab[w]) for w in ok_vocab]
        print("not dict")
        #new_vocab = [w.upper() for w in ok_vocab]
        #print(new_vocab)
        new_vocab = {w.upper(): v for w, v in new_vocab}
        new_vocab = dict(new_vocab)
        #print(new_vocab)




        sections, section = [], None
        wrong_predictions = []
        for line_no, line in enumerate(utils.smart_open(questions)):
            # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
            line = utils.to_unicode(line)
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    self.log_accuracy(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
                try:
                    a, b, c, d, e, expected = [word.upper() for word in line.split()]
                except ValueError:
                    logger.info("skipping invalid line #%i in %s", line_no, questions)
                    continue
                if a not in new_vocab or b not in new_vocab or c not in new_vocab or d not in new_vocab or e not in new_vocab or expected not in new_vocab:
                    #print('not in vocab')
                    logger.debug("skipping line #%i with OOV words: %s", line_no, line.strip())
                    continue

                original_vocab = self.get_vocabulary()
                self.set_vocabulary(new_vocab)
                ignore = {a, b, c, d, e}  # input words to be ignored

                # find the most likely prediction, ignoring OOV words and input words
                sims = self.most_similar(positive_words=[c, d, e], negative_words=[a, b])
                #print("sims")
                #print(sims)
                self.set_vocabulary(original_vocab)

                predicted = sims[0][0]
                predicted = predicted.upper()
                #print(predicted)
                if predicted == expected:
                    section['correct'].append((a, b, c, d, e, expected))
                else:
                    wrong_message = a + " " + b + " " + c + " " + d + " " + e + ", predicted: " + predicted + ", should have been: " + expected
                    section['incorrect'].append((a, b, c, d, e, expected))
                    wrong_predictions.append(wrong_message)
        if section:
            # store the last section, too
            sections.append(section)
            self.log_accuracy(section)

        total = {
            'section': 'total',
            'correct': sum((s['correct'] for s in sections), []),
            'incorrect': sum((s['incorrect'] for s in sections), []),
        }
        self.log_accuracy(total)
        sections.append(total)
        print(wrong_predictions)
        return sections

    def enseemble_results(self, questions):
        """
                Returns a list of the results from an accuracy test

        """
        ok_vocab = self.get_vocabulary()
        new_vocab = [(w, self.model.wv.vocab[w]) for w in ok_vocab]
        new_vocab = {w.upper(): v for w, v in new_vocab}
        new_vocab = dict(new_vocab)

        results = []
        for line_no, line in enumerate(utils.smart_open(questions)):
            # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
            line = utils.to_unicode(line)
            if line.startswith(': '):
                continue
            else:

                try:
                    a, b, c, expected = [word.upper() for word in line.split()]
                except ValueError:
                    logger.info("skipping invalid line #%i in %s", line_no, questions)
                    continue
                if a not in new_vocab or b not in new_vocab or c not in new_vocab or expected not in new_vocab:
                    """if a not in new_vocab:
                        print("Dont know: " + a)
                    if b not in new_vocab:
                        print("Dont know: " + b)
                    if c not in new_vocab:
                        print("Dont know: " + c)
                    if expected not in new_vocab:
                        print("Dont know: " + expected)
                    """
                    logger.debug("skipping line #%i with OOV words: %s", line_no, line.strip())
                    results.append(None)
                    continue

                original_vocab = self.get_vocabulary()
                self.set_vocabulary(new_vocab)
                ignore = {a, b, c}  # input words to be ignored

                # find the most likely prediction, ignoring OOV words and input words
                sims = self.most_similar(positive_words=[b, c], negative_words=[a])
                # print("sims")
                # print(sims)
                self.set_vocabulary(original_vocab)

                predicted = sims[0][0]
                predicted = predicted.upper()

                results.append(predicted)

        return results


