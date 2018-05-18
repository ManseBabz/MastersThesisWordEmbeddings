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

class Stackingmodel:
    def stacking_model_trainer(self, weight_file_name="StackingWeights"):
        savepath = os.path.dirname(os.path.realpath(__file__))+"/LeaningAlgoImpl/Weight_models/"+weight_file_name
        weights = []
        # TODO - Make a way to train the model
        learned_result_to_file = [self.model_list, weights]
        np.save(savepath, learned_result_to_file)
        return learned_result_to_file[1]

    def loaded_stacking_model(self, stacking_model_file_path, positive_word_list, negative_word_list):
        models_and_weights = np.load(stacking_model_file_path)
        if(len(models_and_weights[0])==len(models_and_weights[1])):
            self.model_list=models_and_weights[0]
            self.weight_list=models_and_weights[1]
        else:
            print("Inbalance in weights and models")
            Stackingmodel.stacking_model_trainer(self.model_list)



    def similarity_weighted_avg(self, word_one, word_two):
        results = []
        for model in self.model_list:
            results.append((model.similarity(word_one, word_two)*self.weight_list))
        res = sum(results) / len(results)
        return res

    def accuracy(self, questions, case_insensitive=False, predictor_method=0):
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
                    print("skipping invalid line #%i in %s", line_no, questions)
                    continue
            #TODO predict something
            predicted=[]
            if predicted[0] == expected:
                section['correct'].append((a, b, c, expected))
            else:
                section['incorrect'].append((a, b, c, expected))
        if section:
            # store the last section, too
            sections.append(section)
        total = {
            'section': 'total',
            'correct': sum((s['correct'] for s in sections), []),
            'incorrect': sum((s['incorrect'] for s in sections), []),
        }
        print(total)
        sections.append(total)
        return sections

    def evaluate_word_pairs(self, pairs, delimiter='\t', restrict_vocab=300000,
                            case_insensitive=True, dummy4unknown=False, similarity_model_type="0"):

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
                        a, b, sim = [word.upper() for word in line.split(delimiter)]
                    else:
                        a, b, sim = [word for word in line.split(delimiter)]
                    sim = float(sim)
                except (ValueError, TypeError):
                    #logger.info('skipping invalid line #%d in %s', line_no, pairs)
                    continue

                similarity_gold.append(sim)  # Similarity from the dataset
                similarity_model.append(boot_strap_aggregator.similarity_weighted_avg_proberbility(a, b))


        spearman = stats.spearmanr(similarity_gold, similarity_model)
        pearson = stats.pearsonr(similarity_gold, similarity_model)

        #logger.debug('Pearson correlation coefficient against %s: %f with p-value %f', pairs, pearson[0], pearson[1])
        #logger.debug(
        #    'Spearman rank-order correlation coefficient against %s: %f with p-value %f',
        #    pairs, spearman[0], spearman[1]
        #)
        #logger.debug('Pairs with unknown words: %d', oov)
        #self.log_evaluate_word_pairs(pearson, spearman, oov_ratio, pairs)
        return pearson, spearman


    def __init__(self, model_list):
        self.model_list=model_list