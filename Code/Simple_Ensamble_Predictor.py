import predictor_setup as ps
import os, operator
import numpy as np
from scipy import stats
from Our_FastText import utils
from collections import Counter
from os.path import isfile, join
from os import listdir
import LeaningAlgoImpl.Finished_Models as FM
import numpy.random as random

"""
This model is for playing with, in order to test methods, do not use for testing and alike
"""

####################################################################################################################################
#################################### Helper methods ################################################################################
####################################################################################################################################
def result_unpacker(list_of_results):
    return_result = []
    #print(list_of_results)
    for result_block in list_of_results:
        if(result_block != None):
            for result in result_block:
                return_result.append(result)
    return return_result

def remove_probability_from_result_list(list_of_results):
    real_result = []
    for res in list_of_results:
        real_result.append(res[0])
    return real_result

def update_prob(list_of_results, key_word, prob_value):
    for entry in list_of_results:
        if(entry[0]==key_word):
            entry[1] += prob_value
    return list_of_results



####################################################################################################################################
#################################### Ensamble Methods ##############################################################################
####################################################################################################################################
class simple_ensamble:
    def setup(self):
        self.models = ps.setup(self.model_list, dev_mode=self.dev_mode, training_articles=self.training_articles)

    def predict_majority_vote(self, positive_word_list, negative_word_list, top_n_words, wanted_printed=False):
        result = []  # List of results from the different predictors
        best_result = [None, 0]  # Best result found
        # Predict best word
        for model in self.models:
            result.append(
                model.predict(positive_word_list, negative_word_list))  # Predict from set and add to result list
        result = result_unpacker(result)
        result = remove_probability_from_result_list(result)

        for res in result:
            if (result.count(res) > best_result[1]):  # Check if next result has a better "score"
                best_result = [res, result.count(res)]  # If better score, overwrite best result

        if (wanted_printed == True):
           print(positive_word_list)
           print(negative_word_list)
           print(best_result)
        return best_result

    def predict_sum_proberbility(self, positive_word_list, negative_word_list, top_n_words=1, wanted_printed=False):
        result = []  # List of results from the different predictors
        best_result = [None, 0]  # Best result found
        # Predict best word
        for model in self.models:
            result.append(
                model.predict(positive_word_list, negative_word_list))  # Predict from set and add to result list

        result = result_unpacker(result)
        result = [list(item) for item in result]

        if(result != []):
            true_result_storage = []
            true_result_storage.append(result[0])
            result.pop(0)
            for res in result:
                #print(res)
                for possible in true_result_storage:
                    #print("assessing result")
                    if(res[0]==possible[0]):
                        update_prob(true_result_storage, res[0], res[1])
                        break
                    else:
                        true_result_storage.append(res)
                        break

            best_result = sorted(true_result_storage, key=lambda x: int(x[1]))
            if (wanted_printed == True):
                print(positive_word_list)
                print(negative_word_list)
                print(best_result)
            return best_result[0]
        else:
            return [None]

    def predict_weighted_sum_proberbility(self, positive_word_list, negative_word_list, top_n_words, wanted_printed=False):
        if(self.weight_list == []):
            print("No weights specified, please use: predict_sum_proberbility instead")
            return [None]
        elif(len(self.weight_list) != len(self.models)):
            print("Not enough weights specified")
            return [None]
        else:
            result = []  # List of results from the different predictors
            best_result = [None, 0]  # Best result found
            # Predict best word
            for model_index in range(0, len(self.models)):
                store_res = self.models[model_index].predict(positive_word_list, negative_word_list)  # Predict from set and add to result list
                temp_res = []
                for change_res in store_res:
                    temp_res.append([change_res[0], change_res[1]*self.weight_list[model_index]])
                result.append(temp_res)

            result = result_unpacker(result)
            result = [list(item) for item in result]

            if (result != []):
                true_result_storage = []
                true_result_storage.append(result[0])
                result.pop(0)
                for res in result:
                    # print(res)
                    for possible in true_result_storage:
                        # print("assessing result")
                        if (res[0] == possible[0]):
                            update_prob(true_result_storage, res[0], res[1])
                            break
                        else:
                            true_result_storage.append(res)
                            break

                best_result = sorted(true_result_storage, key=lambda x: int(x[1]))
                if (wanted_printed == True):
                    print(positive_word_list)
                    print(negative_word_list)
                    print(best_result)
                return best_result[0]
            else:
                return [None]

    def similarity_avg_proberbility(self, word_one, word_two):
        results = []
        for model in self.models:
            results.append(model.similarity(word_one, word_two))
        if (all(v is None for v in results)):
            res = 0
        else:
            res = sum(results) / len(results)
        return res

    def similarity_weighted_avg_proberbility(self, word_one, word_two):
        results = []
        for model_index in range(0, len(self.models)):
            res = self.models[model_index].similarity(word_one, word_two)
            if res == None:
                results.append(0)
            else:
                results.append(float(res) * self.weight_list[model_index])
        if (all(v is None for v in results)):
            res = 0
        else:
            res = sum(results) / len(results)
        return res

    def accuracy(self, questions, case_insensitive=False, predictor_method=0):
        correct = 0
        incorrect = 0
        sections, section = [], None
        for line_no, line in enumerate(utils.smart_open(questions)):
            # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
            line = utils.to_unicode(line)
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
                try:
                    if case_insensitive:
                        a, b, c, expected = [word.upper() for word in line.split()]
                    else:
                        a, b, c, expected = [word for word in line.split()]
                except ValueError:
                    #print("skipping invalid line #%i in %s", line_no, questions)
                    continue
                predicted = [None, None]
                if predictor_method == 0:
                    #print("Evaluation method: Majority vote")
                    predicted = simple_ensamble.predict_majority_vote(self, positive_word_list=[b, c],
                                                                        negative_word_list=[a],
                                                                        top_n_words=1)
                elif predictor_method == 1:
                    #print("Evaluation method: Sumed most probable")
                    predicted = simple_ensamble.predict_sum_proberbility(self, positive_word_list=[b, c],
                                                                           negative_word_list=[a],
                                                                           top_n_words=1)
                elif predictor_method == 2:
                    #print("Evaluation method: weighted sum porberbilities")
                    predicted = simple_ensamble.predict_weighted_sum_proberbility(self, positive_word_list=[b, c],
                                                                                    negative_word_list=[a],
                                                                                    top_n_words=1)
                else:
                    raise ValueError("incorrect argument type for predictor_method")

                if predicted[0] == expected:
                    correct += 1
                    section['correct'].append((a, b, c, expected))
                else:
                    incorrect +=1
                    section['incorrect'].append((a, b, c, expected))
        if section:
            # store the last section, too
            sections.append(section)

        print(correct)
        print(incorrect)
        #sections.append(total)
        return sections

    def evaluate_word_pairs(self, pairs, delimiter='\t', restrict_vocab=300000,
                            case_insensitive=False, dummy4unknown=False, similarity_model_type="0"):

        similarity_gold = []
        similarity_model = []
        for line_no, line in enumerate(utils.smart_open(pairs)):
            line = utils.to_unicode(line)
            if line.startswith('#'):
                # May be a comment
                continue
            else:
                try:
                    if case_insensitive:
                        a, b, sim = [word.lower() for word in line.split(delimiter)]
                    else:
                        a, b, sim = [word for word in line.split(delimiter)]
                    sim = float(sim)
                except (ValueError, TypeError):
                    # logger.info('skipping invalid line #%d in %s', line_no, pairs)
                    continue

                similarity_gold.append(sim)  # Similarity from the dataset
                if (similarity_model_type == 0):
                    similarity_model.append(
                        simple_ensamble.similarity_avg_proberbility(self, a, b))  # Similarity from the model
                elif (similarity_model_type == 1):
                    if (self.weight_list == []):
                        raise ValueError("No weights specified for ensamble model")
                    else:
                        similarity_model.append(simple_ensamble.similarity_weighted_avg_proberbility(self, a, b))
                else:
                    raise ValueError("incorrect argument type for predictor_method")

        spearman = stats.spearmanr(similarity_gold, similarity_model)
        pearson = stats.pearsonr(similarity_gold, similarity_model)

        # logger.debug('Pearson correlation coefficient against %s: %f with p-value %f', pairs, pearson[0], pearson[1])
        # logger.debug(
        #    'Spearman rank-order correlation coefficient against %s: %f with p-value %f',
        #    pairs, spearman[0], spearman[1]
        # )
        # logger.debug('Pairs with unknown words: %d', oov)
        # self.log_evaluate_word_pairs(pearson, spearman, oov_ratio, pairs)
        print(pearson)
        print(spearman)
        return pearson, spearman

    def set_weights(self, weight_list):
        self.weight_list = weight_list

    def __init__(self, model_name_list, dev_mode=False, training_articles=10000):
        self.weight_list = []
        self.models = []
        self.model_list=model_name_list
        self.dev_mode = dev_mode
        self.training_articles = training_articles
        simple_ensamble.setup(self)


