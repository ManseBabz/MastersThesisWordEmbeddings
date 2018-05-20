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

class boot_strap_aggregator:
    def setup(self):
        if(len(self.model_list)<5):
            self.models = ps.setup(self.model_list, dev_mode=self.dev_mode, training_articles=self.training_articles)
        else:
            self.large_model = True

    def predict_majority_vote(self, positive_word_list, negative_word_list, top_n_words, wanted_printed=False):
        result = []  # List of results from the different predictors
        best_result = [None, 0]  # Best result found
        # Predict best word
        if(self.large_model == False):
            for model in self.models:
                result.append(
                    model.predict(positive_word_list, negative_word_list))  # Predict from set and add to result list
            result = result_unpacker(result)
            result = remove_probability_from_result_list(result)
            for res in result:
                if (result.count(res) > best_result[1]):  # Check if next result has a better "score"
                    best_result = [res, result.count(res)]  # If better score, overwrite best result
        else:
            for model_string in self.model_list:
                temp_model_string = [model_string]
                temp_model = ps.setup(temp_model_string, self.dev_mode)
                for temp in temp_model:
                    result.append(temp.predict(positive_word_list, negative_word_list))
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
        results=[]
        for model in self.models:
            results.append(model.similarity(word_one, word_two))
        if(all(v is None for v in results)):
            res = 0
        else:
            res = sum(results)/len(results)
        return res

    def similarity_weighted_avg_proberbility(self, word_one, word_two):
        results = []
        for model_index in range(0, len(self.models)):
            res =self.models[model_index].similarity(word_one, word_two)
            if res == None:
                results.append(0)
            else:
                results.append(float(res)*self.weight_list[model_index])
        if (all(v is None for v in results)):
            res = 0
        else:
            res = sum(results) / len(results)
        return res

    def majority_vote_fast(self, questions, topn=1, number_of_models = 20):
        guesses = self.get_multiple_results(questions, number_of_models=number_of_models, topn=topn)
        reals = self.get_expected_acc_results(questions)

        combined_guesses = []
        for i in range(len(reals)):
            combined_guess = []
            for guess in guesses:
                #print(guess[i])

                try:
                    for g in guess[i]:
                        combined_guess.append(g)
                except TypeError:
                    combined_guess.append(guess[i])

                # combined_guess.append(guess[i])
            combined_guesses.append(combined_guess)

        # print(combined_guesses)
        # print(reals)

        final_guess = []
        for guess in combined_guesses:
            count = Counter(guess)
            most_common = count.most_common(1)[0]
            #print(most_common)
            if (most_common[0] is None):
                # print('hej')
                try:
                    most_common = count.most_common(2)[1]
                except IndexError:
                    #print('No model has an answer')
                    pass

            final_guess.append(most_common[0])
        #print(final_guess)

        correct = []
        wrong = []
        number_of_correct = 0
        number_of_wrong = 0
        for i in range(0, len(final_guess)):
            predicted = final_guess[i]
            expected = reals[i]
            if predicted == expected:
                correct_message = ("predicted: %s correct" % (predicted))
                correct.append(correct_message)
                number_of_correct += 1
            else:
                wrong_message = ("predicted: %s, should have been: %s" % (predicted, expected))
                wrong.append(wrong_message)
                number_of_wrong += 1

        print('Correct ' + str(number_of_correct))
        #print(correct)
        print('Wrong ' + str(number_of_wrong))
        #print(wrong)
        return number_of_correct, number_of_wrong

    def get_multiple_results(self, questions, topn, number_of_models=5):
        name_array = []
        not_wanted = ['npy', 'Readme.md']
        onlyfiles = [f for f in listdir(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models") if
                     isfile(join(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models", f))]
        for file_index in range(0, len(onlyfiles)):
            if (not_wanted[0] in onlyfiles[file_index] or not_wanted[1]in onlyfiles[file_index]):
                continue
            else:
                name_array.append(onlyfiles[file_index])

        #print(name_array)
        models = []
        random.shuffle(name_array)
        #print(name_array)
        i = 0
        for name in name_array:
            if number_of_models > i:
                print(name)
                finished_model = FM.Finished_Models()
                finished_model.get_model(
                    os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models/" + name)
                models.append(finished_model)
                i += 1
            else:
                continue

        #print(models)

        results = []
        for model in models:
            results.append(model.get_acc_results(topn, questions))

        #print(results)
        return results


    def position_based_tie_handling_majority_vote(self, questions, topn = 1, number_of_models=10):
        guesses = self.get_multiple_results(questions, number_of_models=number_of_models, topn=topn)
        reals = self.get_expected_acc_results(questions)

        combined_guesses = []
        for i in range(len(reals)):
            combined_guess = []
            for guess in guesses:
                #print(guess[i])

                try:
                    for j in range(0, len(guess[i])):
                        combined_guess.append((guess[i][j], j))
                except TypeError:
                    combined_guess.append(guess[i])

                #combined_guess.append(guess[i])
            combined_guesses.append(combined_guess)

        #print(combined_guesses)
        #print(reals)

        final_guess = []
        for guess in combined_guesses:
            count = self.position_counter(guess)
            most_common = count[0]
            #print(most_common)
            if(most_common[0] is None):
                #print('hej')
                try:
                    most_common = count[1]
                except IndexError:
                    #print('No model has an answer')
                    pass
            #print(most_common[0])
            final_guess.append(most_common[0])
        #print(final_guess)

        correct = []
        wrong = []
        number_of_correct = 0
        number_of_wrong = 0
        for i in range(0, len(final_guess)):
            predicted = final_guess[i]
            expected = reals[i]
            if predicted == expected:
                correct_message = ("predicted: %s correct" % (predicted))
                correct.append(correct_message)
                number_of_correct += 1
            else:
                wrong_message = ("predicted: %s, should have been: %s" % (predicted, expected))
                wrong.append(wrong_message)
                number_of_wrong += 1

        print('Correct ' + str(number_of_correct))
        print(correct)
        print('Wrong ' + str(number_of_wrong))
        print(wrong)
        return number_of_correct, number_of_wrong

    def position_counter(self, guess_list):
        # print('guess')
        #print(guess_list)
        combined_guess_list = []
        # print(guess_list[0])

        # if guess_list is None:
        # return [[None]]
        for i in range(0, len(guess_list)):
            if guess_list[i] is not None:
                guess, value = guess_list[i]
                not_found = True
                for i in range(0, len(combined_guess_list)):
                    word, old_value, number_of_appearances = combined_guess_list[i]
                    if guess == word:
                        combined_guess_list[i] = [guess, value + old_value, number_of_appearances + 1]
                        not_found = False

                if not_found:
                    combined_guess_list.append((guess, value, 1))
            else:
                # print('None')
                combined_guess_list.append((None, 0, 0))
                # print(combined_guess_list)

        sorted_combined_guess_list = sorted(combined_guess_list, key=lambda x: x[2], reverse=True)
        #print(sorted_combined_guess_list)
        max_number_of_appearance = []
        for i in range(0, len(sorted_combined_guess_list)):
            #print(sorted_combined_guess_list[i])
            if sorted_combined_guess_list[0][2] == sorted_combined_guess_list[i][2]:
                max_number_of_appearance.append((sorted_combined_guess_list[i][0], sorted_combined_guess_list[i][1]))
        sorted_combined_guess_list = sorted(max_number_of_appearance, key=lambda x: x[1], reverse=False)
        #print(sorted_combined_guess_list)
        return sorted_combined_guess_list





    def tie_handling_majority_vote(self, questions, topn = 1, number_of_models=10):
        guesses = self.get_multiple_tie_handling_results(questions, number_of_models=number_of_models, topn=topn)
        reals = self.get_expected_acc_results(questions)

        #print(reals)
        print(guesses)
        combined_guesses = []
        for i in range(len(reals)):
            combined_guess = []
            for guess in guesses:
                #print(guess[i])

                try:
                    for g in guess[i]:
                        combined_guess.append(g)
                except TypeError:
                    combined_guess.append(guess[i])

                #combined_guess.append(guess[i])
            combined_guesses.append(combined_guess)

        #print(combined_guesses)
        #print(reals)

        final_guess = []
        for guess in combined_guesses:

            count = self.weighted_counter(guess)

            most_common = count[0]
            #print(most_common)
            if(most_common[0] is None):
                #print('hej')
                try:
                    most_common = count[1]
                except IndexError:
                    #print('No model has an answer')
                    pass
            #print(most_common[0])
            final_guess.append(most_common[0])
        #print(final_guess)

        correct = []
        wrong = []
        number_of_correct = 0
        number_of_wrong = 0
        for i in range(0, len(final_guess)):
            predicted = final_guess[i]
            expected = reals[i]
            if predicted == expected:
                correct_message = ("predicted: %s correct" % (predicted))
                correct.append(correct_message)
                number_of_correct += 1
            else:
                wrong_message = ("predicted: %s, should have been: %s" % (predicted, expected))
                wrong.append(wrong_message)
                number_of_wrong += 1

        print('Correct ' + str(number_of_correct))
        #print(correct)
        print('Wrong ' + str(number_of_wrong))
        #print(wrong)
        return number_of_correct, number_of_wrong

    def weighted_counter(self, guess_list):
        #print('guess')
        #print(guess_list)
        combined_guess_list = []
        #print(guess_list[0])

        #if guess_list is None:
            #return [[None]]
        for i in range(0, len(guess_list)):
            if guess_list[i] is not None:
                guess, value = guess_list[i]
                not_found = True
                for i in range(0, len(combined_guess_list)):
                    word, old_value = combined_guess_list[i]
                    if guess == word:
                        combined_guess_list[i] = (guess, value + old_value)
                        not_found = False

                if not_found:
                    combined_guess_list.append((guess, value))
            else:
                #print('None')
                combined_guess_list.append((None, 0))
            #print(combined_guess_list)

        sorted_combined_guess_list = sorted(combined_guess_list, key=lambda x: x[1], reverse=True)
        #print(sorted_combined_guess_list)
        return sorted_combined_guess_list





    def get_multiple_tie_handling_results(self, questions, number_of_models, topn):
        name_array = []
        not_wanted = ['npy', 'Readme.md']
        onlyfiles = [f for f in listdir(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models") if
                     isfile(join(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models", f))]
        for file_index in range(0, len(onlyfiles)):
            if (not_wanted[0] in onlyfiles[file_index] or not_wanted[1]in onlyfiles[file_index]):
                continue
            else:
                name_array.append(onlyfiles[file_index])

        #print(name_array)
        random.shuffle(name_array)
        results = []
        i = 0
        for name in name_array:
            if number_of_models > i:
                print(name)
                finished_model = FM.Finished_Models()
                finished_model.get_model(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models/" + name)
                results.append(finished_model.get_acc_results_extra(topn, questions))
                i += 1
            else:
                continue

        #print(results)
        return results








    def get_expected_acc_results(self, questions):
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        questions = dir_path + '/Code/TestingSet/' + questions
        """
                Returns a list of the expected results from an accuracy test

        """
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

                results.append(expected)
        return results

    def get_acc_results(self, model, questions):
        return model.get_acc_results(questions)









    def accuracy(self, questions, case_insensitive=True, predictor_method=0, fast_process=True, number_of_models=20):
        correct = 0
        incorrect = 0
        sections, section = [], None
        if(fast_process == False):
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
                        #print("Evaluation method: Majority vote")
                        predicted = boot_strap_aggregator.predict_majority_vote(self, positive_word_list=[b, c],
                                                                            negative_word_list=[a],
                                                                            top_n_words=1)
                    elif predictor_method == 1:
                        #print("Evaluation method: Sumed most probable")
                        predicted = boot_strap_aggregator.predict_sum_proberbility(self, positive_word_list=[b, c],
                                                                               negative_word_list=[a],
                                                                               top_n_words=1)
                    elif predictor_method == 2:
                        #print("Evaluation method: weighted sum porberbilities")
                        predicted = boot_strap_aggregator.predict_weighted_sum_proberbility(self, positive_word_list=[b, c],
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
        else:
            if(predictor_method== 0):
                correct, wrong = boot_strap_aggregator.majority_vote_fast(self, questions=questions, topn=10, number_of_models=number_of_models)
                return correct, wrong
            elif(predictor_method==3):
                correct, wrong = boot_strap_aggregator.tie_handling_majority_vote(self, questions=questions, topn=10,
                                                                          number_of_models=number_of_models)
                return correct, wrong
            elif(predictor_method==4):
                correct, wrong = boot_strap_aggregator.position_based_tie_handling_majority_vote(self, questions=questions, topn=10, number_of_models=number_of_models)
                return correct, wrong

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
                    #logger.info('skipping invalid line #%d in %s', line_no, pairs)
                    continue

                similarity_gold.append(sim)  # Similarity from the dataset
                if(similarity_model_type == 0):
                    similarity_model.append(boot_strap_aggregator.similarity_avg_proberbility(self, a, b))  # Similarity from the model
                elif(similarity_model_type == 1):
                    if(self.weight_list == []):
                        raise ValueError("No weights specified for ensamble model")
                    else:
                        similarity_model.append(boot_strap_aggregator.similarity_weighted_avg_proberbility(self, a, b))
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
        print(pearson)
        print(spearman)
        return pearson, spearman

    def set_weights(self, weight_list):
        self.weight_list = weight_list

    def __init__(self, model_name_list = [], dev_mode=False, training_articles=10000):
        self.models = []
        self.model_list=model_name_list
        self.dev_mode = dev_mode
        self.training_articles = training_articles
        self.large_model = False
        boot_strap_aggregator.setup(self)