import LeaningAlgoImpl.Finished_Models as FM
import logging, os, time
from os import listdir
from os.path import isfile, join
from gensim import utils
from collections import Counter

class own_enseemble:

    def majority_vote(self, questions, topn = 1):
        guesses = self.get_multiple_results(questions, number_of_models=9, topn=topn)
        reals = self.get_expected_acc_results(questions)

        combined_guesses = []
        for i in range(len(reals)):
            combined_guess = []
            for guess in guesses:
                print(guess[i])

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
            print(most_common)
            if(most_common[0] is None):
                #print('hej')
                try:
                    most_common = count.most_common(2)[1]
                except IndexError:
                    print('No model has an answer')
                    pass

            final_guess.append(most_common[0])
        print(final_guess)

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

        print(name_array)
        models = []
        i = 0
        for name in name_array:
            if number_of_models > i:
                print(name)
                finished_model = FM.Finished_Models()
                finished_model.get_model(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models/" + name)
                models.append(finished_model)
                i += 1
            else:
                continue

        print(models)

        results = []
        for model in models:
            results.append(model.get_acc_results(topn, questions))

        print(results)
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

enseemble_test.majority_vote('danish-topology.txt', topn=10)