import LeaningAlgoImpl.Finished_Models as FM
import logging, os, time
from os import listdir
from os.path import isfile, join
from gensim import utils
from collections import Counter
import numpy.random as random
from scipy import stats

class own_enseemble:

    def simple_majority_vote(self, questions, topn = 1):
        guesses = self.get_multiple_results(questions, number_of_models=15, topn=topn)
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

                #combined_guess.append(guess[i])
            combined_guesses.append(combined_guess)

        #print(combined_guesses)
        #print(reals)

        final_guess = []
        for guess in combined_guesses:
            count = Counter(guess)
            most_common = count.most_common(1)[0]
            #print(most_common)
            if(most_common[0] is None):
                #print('hej')
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
        print(correct)
        print('Wrong ' + str(number_of_wrong))
        print(wrong)

    def position_based_tie_handling_majority_vote(self, questions, topn = 1):
        guesses = self.get_multiple_results(questions, number_of_models=15, topn=topn)
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

    def get_multiple_results(self, questions, number_of_models, topn):
        name_array = []
        not_wanted = 'npy'
        onlyfiles = [f for f in listdir(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models") if
                     isfile(join(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models", f))]
        for file_index in range(0, len(onlyfiles)):
            if (not_wanted in onlyfiles[file_index]):
                continue
            else:
                name_array.append(onlyfiles[file_index])

        #print(name_array)
        models = []
        i = 0
        for name in name_array:
            if number_of_models > i:
                #print(name)
                finished_model = FM.Finished_Models()
                finished_model.get_model(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models/" + name)
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

    def certainess_tie_handling_majority_vote(self, questions, topn = 1):
        guesses = self.get_multiple_certainess_tie_handling_results(questions, number_of_models=15, topn=topn)
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
        print(correct)
        print('Wrong ' + str(number_of_wrong))
        print(wrong)

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





    def get_multiple_certainess_tie_handling_results(self, questions, number_of_models, topn):
        name_array = []
        not_wanted = 'npy'
        onlyfiles = [f for f in listdir(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models") if
                     isfile(join(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models", f))]
        for file_index in range(0, len(onlyfiles)):
            if (not_wanted in onlyfiles[file_index]):
                continue
            else:
                name_array.append(onlyfiles[file_index])

        #print(name_array)
        models = []
        i = 0
        for name in name_array:
            if number_of_models > i:
                #print(name)
                finished_model = FM.Finished_Models()
                finished_model.get_model(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models/" + name)
                models.append(finished_model)
                i += 1
            else:
                continue

        #print(models)

        results = []
        for model in models:
            results.append(model.get_acc_results_extra(topn, questions))

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

    def naive_human_similarity_majority_vote(self, questions, number_of_models):
        reals = self.get_human_similarities_results(questions)
        guesses = self.get_model_similarities_results(questions, number_of_models)

        combined_guesses = []
        for j in range(0, len(guesses[0][0])):
            combined_guess = []
            for i in range(0, number_of_models):
                combined_guess.append(guesses[i][0][j])
            #print(combined_guess)
            combined_guesses.append(combined_guess)
        #print(combined_guesses)

        average_guesses = []
        for guess in combined_guesses:
            average_guess = sum(guess)/len(guess)
            average_guesses.append(average_guess)

        print(len(reals))
        print(reals)
        print(len(average_guesses))
        print(average_guesses)

        spearman_result = stats.spearmanr(reals, average_guesses)
        pearson_result = stats.pearsonr(reals, average_guesses)
        return spearman_result, pearson_result


    def get_human_similarities_results(self, test_set):
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        test_set = dir_path + '/Code/TestingSet/' + test_set
        results = []
        for line_no, line in enumerate(utils.smart_open(test_set)):
            line = utils.to_unicode(line)
            if line.startswith('#'):
                # May be a comment
                continue
            else:
                try:

                    a, b, sim = [word.upper() for word in line.split('\t')]

                    sim = float(sim)
                except (ValueError, TypeError):
                    logger.info('skipping invalid line #%d in %s', line_no, test_set)
                    continue

                results.append(sim)  # Similarity from the dataset
        return results

    def get_model_similarities_results(self, questions, number_of_models):
        name_array = []
        not_wanted = ['npy', 'Readme.md']
        onlyfiles = [f for f in listdir(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models") if
                     isfile(join(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models", f))]
        for file_index in range(0, len(onlyfiles)):
            if (not_wanted[0] in onlyfiles[file_index] or not_wanted[1] in onlyfiles[file_index]):
                continue
            else:
                name_array.append(onlyfiles[file_index])

        # print(name_array)
        random.shuffle(name_array)
        results = []
        i = 0
        for name in name_array:
            if number_of_models > i:
                print(name)
                finished_model = FM.Finished_Models()
                finished_model.get_model(
                    os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models/" + name)
                results.append(finished_model.model_similarity_results(questions))
                i += 1
            else:
                continue

        # print(results)
        return results


    def __init__(self):
        print("init")



dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

finished_model = FM.Finished_Models()
finished_model.get_model(dir_path + '/Code/LeaningAlgoImpl/Models/CBOW,0,5,0,10,100,1,90000_Trained_on1000000articles')

enseemble_test = own_enseemble()
#res = enseemble_test.get_acc_results(finished_model, 'danish-topology.txt')

#print(res)

#enseemble_test.get_multiple_results('danish-topology.txt', 4)
#print(enseemble_test.get_expected_acc_results('danish-topology.txt'))

#enseemble_test.simple_majority_vote('danish-topology.txt', topn=10)

#enseemble_test.certainess_tie_handling_majority_vote('danish-topology.txt', topn=10)

#enseemble_test.position_based_tie_handling_majority_vote('danish-topology.txt', topn=10)

print(enseemble_test.get_human_similarities_results('wordsim353.tsv'))
#print(enseemble_test.get_model_similarities_results('wordsim353.tsv', 10))
print(enseemble_test.naive_human_similarity_majority_vote('wordsim353.tsv', 10))
