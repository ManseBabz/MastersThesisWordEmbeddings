import predictor_setup as ps
import os, operator
import numpy as np
from scipy import stats
from Our_FastText import utils
"""
This module contains multiple ensamble methods
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

def contains(list_of_results, keyword):
    truth_value = False
    for string in list_of_results:
        if(string == keyword):
            truth_value =True
    return truth_value

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

    def predict_sum_proberbility(self, positive_word_list, negative_word_list, top_n_words, wanted_printed=False):
        result = []  # List of results from the different predictors
        best_result = [None, 0]  # Best result found

        for model in self.models:
            result.append(
                model.predict(positive_word_list, negative_word_list))  # Predict from set and add to result list

        result = result_unpacker(result)
        most_probable_result_storing = []
        for res in result:
            if (contains(most_probable_result_storing, res[0])):
                update_prob(most_probable_result_storing, res[0], res[1])
            else:
                most_probable_result_storing.append(res)
        most_probable_result_storing.sort(key=operator.itemgetter(1), reverse=True)
        best_result = most_probable_result_storing[0]

        if (wanted_printed == True):
            print(positive_word_list)
            print(negative_word_list)
            print(most_probable_result_storing)
            print(best_result)
        return best_result

    def predict_weighted_sum_proberbility(self, positive_word_list, negative_word_list, top_n_words, wanted_printed=False):
        result = []  # List of results from the different predictors
        best_result = [None, 0]  # Best result found
        # Predict best word and calculate weightet probability for this word
        for i in range(0, len(self.models)):
            res = self.models[i].predict(positive_word_list=positive_word_list,
                                    negative_word_list=negative_word_list)  # Predict from set and add to result list
            for part_res in res:
                temp_res = []
                temp_res.append(part_res[0])
                temp_res.append(part_res[1] * self.weight_list[i])
                result.append(temp_res)

        # Combine probability for results
        most_probable_result_storing = []
        for res in result:
            if (contains(most_probable_result_storing, res[0])):
                update_prob(most_probable_result_storing, res[0], res[1])
            else:
                most_probable_result_storing.append(res)
        most_probable_result_storing.sort(key=operator.itemgetter(1), reverse=True)

        # Pick best result
        if (most_probable_result_storing != []):
            if (wanted_printed == True):
                print(positive_word_list)
                print(negative_word_list)
                print(most_probable_result_storing[0])
            return most_probable_result_storing[0]
        else:
            print("No result")
            return []

    def similarity_avg(self, word_one, word_two):
        results=[]
        for model in self.model_list:
            results.append(model.similarity(word_one, word_two))
        res = sum(results)/len(results)
        return res

    def similarity_weighted_avg(self, word_one, word_two):
        results = []
        for model in self.model_list:
            results.append((model.similarity(word_one, word_two)*self.weight_list))
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
                    print("skipping invalid line #%i in %s", line_no, questions)
                    continue
                predicted = [None, None]
                if predictor_method == 0:
                    print("Evaluation method: Majority vote")
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
                if(similarity_model_type == 0):
                    similarity_model.append(boot_strap_aggregator.similarity_avg_proberbility(a, b))  # Similarity from the model
                elif(similarity_model_type == 1):
                    if(self.weight_list == []):
                        raise ValueError("No weights specified for ensamble model")
                    else:
                        similarity_model.append(boot_strap_aggregator.similarity_weighted_avg_proberbility(a, b))
                else:
                    raise ValueError("incorrect argument type for predictor_method")

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

    def set_weights(self, weight_list):
        self.weight_list = weight_list

    def __init__(self, model_name_list, dev_mode=False, training_articles=10000):
        self.models = []
        self.model_list=model_name_list
        self.dev_mode = dev_mode
        self.training_articles = training_articles
        simple_ensamble.setup(self)


"""
    Bootstrap aggregation model
    
    This model dosn't have a development mode due to the random sampeling nature of the ensamble method
"""
class boot_strap_aggregator:
    def setup(self):
        self.models = ps.setup(self.model_list, training_articles=self.training_articles)

    def predict_majority_vote(self, positive_word_list, negative_word_list, top_n_words, wanted_printed=False):
        result = []  # List of results from the different predictors
        best_result = [None, 0]  # Best result found
        # Predict best word
        for model in self.models:
            result.append(
                model.predict(positive_word_list, negative_word_list, nwords=top_n_words))  # Predict from set and add to result list
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

    def predict_sum_proberbility(self, positive_word_list, negative_word_list, top_n_words, wanted_printed=False):
        result = []  # List of results from the different predictors
        best_result = [None, 0]  # Best result found

        for model in self.models:
            result.append(
                model.predict(positive_word_list, negative_word_list, nwords=top_n_words))  # Predict from set and add to result list

        result = result_unpacker(result)
        most_probable_result_storing = []
        for res in result:
            if (contains(most_probable_result_storing, res[0])):
                update_prob(most_probable_result_storing, res[0], res[1])
            else:
                most_probable_result_storing.append(res)
        most_probable_result_storing.sort(key=operator.itemgetter(1), reverse=True)
        best_result = most_probable_result_storing[0]

        if (wanted_printed == True):
            print(positive_word_list)
            print(negative_word_list)
            print(most_probable_result_storing)
            print(best_result)
        return best_result

    def predict_weighted_sum_proberbility(self, positive_word_list, negative_word_list, top_n_words, wanted_printed=False):
        result = []  # List of results from the different predictors
        best_result = [None, 0]  # Best result found
        # Predict best word and calculate weightet probability for this word
        for i in range(0, len(self.models)):
            res = self.models[i].predict(positive_word_list=positive_word_list,
                                    negative_word_list=negative_word_list)  # Predict from set and add to result list
            for part_res in res:
                temp_res = []
                temp_res.append(part_res[0])
                temp_res.append(part_res[1] * self.weight_list[i])
                result.append(temp_res)

        # Combine probability for results
        most_probable_result_storing = []
        for res in result:
            if (contains(most_probable_result_storing, res[0])):
                update_prob(most_probable_result_storing, res[0], res[1])
            else:
                most_probable_result_storing.append(res)
        most_probable_result_storing.sort(key=operator.itemgetter(1), reverse=True)

        # Pick best result
        if (most_probable_result_storing != []):
            if (wanted_printed == True):
                print(positive_word_list)
                print(negative_word_list)
                print(most_probable_result_storing[0])
            return most_probable_result_storing[0]
        else:
            print("No result")
            return []

    def similarity_avg(self, word_one, word_two):
        results=[]
        for model in self.model_list:
            results.append(model.similarity(word_one, word_two))
        res = sum(results)/len(results)
        return res

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
            if predictor_method == 0:
                print("Evaluation method: Majority vote")
                predicted = boot_strap_aggregator.predict_majority_vote(positive_word_list=[b, c], negative_word_list=[a],
                                                            top_n_words=1)
            elif predictor_method ==1:
                print("Evaluation method: Sumed most probable")
                predicted = boot_strap_aggregator.predict_sum_proberbility(positive_word_list=[b, c], negative_word_list=[a],
                                                            top_n_words=1)
            elif predictor_method ==2:
                print("Evaluation method: weighted sum porberbilities")
                predicted = boot_strap_aggregator.predict_weighted_sum_proberbility(positive_word_list=[b, c], negative_word_list=[a],
                                                            top_n_words=1)
            else:
                raise ValueError("incorrect argument type for predictor_method")


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
                if(similarity_model_type == 0):
                    similarity_model.append(boot_strap_aggregator.similarity_avg_proberbility(a, b))  # Similarity from the model
                elif(similarity_model_type == 1):
                    if(self.weight_list == []):
                        raise ValueError("No weights specified for ensamble model")
                    else:
                        similarity_model.append(boot_strap_aggregator.similarity_weighted_avg_proberbility(a, b))
                else:
                    raise ValueError("incorrect argument type for predictor_method")

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

    def set_weights(self, weight_list):
        self.weight_list = weight_list

    def __init__(self, model_name_list, training_articles=10000):
        self.model_list=model_name_list
        self.training_articles = training_articles
        boot_strap_aggregator.setup(model_name_list)


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
            if Stackingmodel == 0:
                print("Evaluation method: Majority vote")
                predicted = boot_strap_aggregator.predict_majority_vote(positive_word_list=[b, c],
                                                                        negative_word_list=[a],
                                                                        top_n_words=1)
            elif Stackingmodel == 1:
                print("Evaluation method: Sumed most probable")
                predicted = boot_strap_aggregator.predict_sum_proberbility(positive_word_list=[b, c],
                                                                           negative_word_list=[a],
                                                                           top_n_words=1)
            elif Stackingmodel == 2:
                print("Evaluation method: weighted sum porberbilities")
                predicted = boot_strap_aggregator.predict_weighted_sum_proberbility(positive_word_list=[b, c],
                                                                                    negative_word_list=[a],
                                                                                    top_n_words=1)
            else:
                raise ValueError("incorrect argument type for predictor_method")

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
####################################################################################################################################
#################################### Main method for testing pourpuse ##############################################################
####################################################################################################################################

"""
    The simple majority vote ensamble takes a 3 level array i.e. a tensor.
    The outer layer for defining what type of word-embedding to run. 
    
    Array arguments
    0 - hs=hs, #1 for hierarchical softmax and 0 and non-zero in negative argument then negative sampling is used.
    1 - negative=negative, #0 for no negative sampling and above specifies how many noise words should be drawn. (Usually 5-20 is good).
    2 - cbow_mean=cbow_mean, #0 for sum of context vectors, 1 for mean of context vectors. Only used on CBOW.
    3 - iter=iter, #number of epochs.
    4 - size=size, #feature vector dimensionality
    5 - min_count=min_count, #minimum frequency of words required
    6 - max_vocab_size=max_vocab_size, #How much RAM is allowed, 10 million words needs approx 1GB RAM. None = infinite RAM
    7 - workers=workers, #How many threads are started for training.
    
    
    For special_fast_text
    hs=i[0], negative=i[1], iter=i[2], size=i[3], min_count=i[4],
                              max_vocab_size=i[5], workers=i[6], min_n=i[7], max_n=i[8], adaptive=i[9], word_ngrams=i[10]  
"""
