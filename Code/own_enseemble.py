import LeaningAlgoImpl.Finished_Models as FM
import logging, os, time
from os import listdir
from os.path import isfile, join
from gensim import utils
from collections import Counter
import numpy.random as random
from scipy import stats
import csv

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

    def ignore_oov_human_similarity_majority_vote(self, questions, number_of_models):
        reals = self.get_human_similarities_results(questions)
        guesses = self.get_model_similarities_results(questions, number_of_models)

        combined_guesses = []
        for j in range(0, len(guesses[0][0])):
            combined_guess = []
            for i in range(0, number_of_models):
                if guesses[i][0][j] == 0.0:
                    continue
                combined_guess.append(guesses[i][0][j])
            # print(combined_guess)
            combined_guesses.append(combined_guess)
        # print(combined_guesses)

        average_guesses = []
        for guess in combined_guesses:
            print(len(guess))
            if len(guess) == 0:
                average_guesses.append(0.0)
            else:
                average_guess = sum(guess) / len(guess)
                average_guesses.append(average_guess)

        print(len(reals))
        print(reals)
        print(len(average_guesses))
        print(average_guesses)

        spearman_result = stats.spearmanr(reals, average_guesses)
        pearson_result = stats.pearsonr(reals, average_guesses)
        return spearman_result, pearson_result

    def weight_based_on_oov_human_similarity_majority_vote(self, questions, number_of_models):
        reals = self.get_human_similarities_results(questions)
        guesses = self.get_model_similarities_results(questions, number_of_models)
        #print(guesses)
        guesses = sorted(guesses, key=lambda x: x[1], reverse=True) # changed
        #print(guesses)
        combined_guesses = []
        for j in range(0, len(guesses[0][0])):
            combined_guess = []
            for i in range(0, number_of_models):
                combined_guess.append((guesses[i][0][j], i + 1)) # changed
            #print(combined_guess)
            combined_guesses.append(combined_guess)
        #print(combined_guesses)

        average_guesses = []
        for guess in combined_guesses:
            combined_oov = 0
            for g in guess:
                combined_oov += g[1]
            weighted_guess = 0
            for g in guess:
                weighted_guess += g[0] * (g[1] / combined_oov)

            average_guesses.append(weighted_guess)

        print(len(reals))
        print(reals)
        print(len(average_guesses))
        print(average_guesses)

        spearman_result = stats.spearmanr(reals, average_guesses)
        pearson_result = stats.pearsonr(reals, average_guesses)
        return spearman_result, pearson_result

    def weight_based_on_total_oov_ignore_oov_human_similarity_majority_vote(self, questions, number_of_models):
        reals = self.get_human_similarities_results(questions)
        guesses = self.get_model_similarities_results(questions, number_of_models)

        guesses = sorted(guesses, key=lambda x: x[1], reverse=True)

        combined_guesses = []
        for j in range(0, len(guesses[0][0])):
            combined_guess = []
            for i in range(0, number_of_models):
                if guesses[i][0][j] == 0.0:
                    continue
                combined_guess.append((guesses[i][0][j], i + 1))
            print(combined_guess)
            combined_guesses.append(combined_guess)
            # print(combined_guesses)

        average_guesses = []
        for guess in combined_guesses:
            if len(guess) == 0:
                average_guesses.append(0.0)
                continue
            combined_oov = 0
            for g in guess:
                combined_oov += g[1]
            weighted_guess = 0
            for g in guess:
                weighted_guess += g[0] * (g[1] / combined_oov)

            average_guesses.append(weighted_guess)

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

    def naive_clustering_majority_vote(self, test_set, number_of_models):
        test_set = dir_path + '/Code/TestingSet/' + test_set

        reals = []
        with open(test_set) as csvfile:
            test_reader = csv.reader(csvfile, delimiter=',')
            #initialize first cluster
            cluster = []
            for row in test_reader:
                if not row:
                    #Add new cluster
                    reals.append(cluster)
                    cluster = []
                else:
                    cluster.append(''.join(row))
            # add last cluster
            reals.append(cluster)

        print(reals)

        guesses = self.get_models_clustering_results(reals, number_of_models)

        print(guesses)
        #print(guesses[0][0])

        # Assign first cluster to final cluster
        final_clustering = guesses[0][0]

        for guess in guesses:
            new_final_clustering = []

            for i in range(0, len(guess[1])):
                for cluster in final_clustering:
                    if guess[1][i] in cluster:
                        clust = list(cluster)

                        for word in guess[0][i]:
                            if word not in clust:
                                clust.append(word)
                        new_final_clustering.append(clust)


            for cluster in final_clustering:
                for clust in new_final_clustering:
                    if clust[0] in cluster:
                        for word in clust:
                            if word not in cluster:
                                cluster.append(word)



        final_clustering.sort(key=len)

        correct, wrong = self.get_clustering_result(final_clustering, reals)

        return correct, wrong

    def get_biggest_first_clustering_majority_vote(self, test_set, number_of_models):
        test_set = dir_path + '/Code/TestingSet/' + test_set

        reals = []
        with open(test_set) as csvfile:
            test_reader = csv.reader(csvfile, delimiter=',')
            #initialize first cluster
            cluster = []
            for row in test_reader:
                if not row:
                    #Add new cluster
                    reals.append(cluster)
                    cluster = []
                else:
                    cluster.append(''.join(row))
            # add last cluster
            reals.append(cluster)
        print(reals)


        guesses = self.get_models_clustering_results(reals, number_of_models)

        #print(guesses)

        clusters = []
        mediods = []
        for guess in guesses:
            #print(guess)
            #print(guess[1])
            clusters.append(guess[0])
            mediods.append(guess[1])
        #print(clusters)
        #print(mediods)

        cluster_with_len = []

        for cluster in clusters:
            number_of_results = 0
            for clust in cluster:
                number_of_results += len(clust)
            cluster_with_len.append((cluster, number_of_results))
        #print(cluster_with_len)

        cluster_with_len.sort(key=lambda x: x[1], reverse=True)

        #print(cluster_with_len)
        #print(cluster_with_len[0][0])

        final_clustering = cluster_with_len[0][0]

        for guess in cluster_with_len:
            new_final_clustering = []
            #print(guess)
            #print(guess[0])
            for i in range(0, len(guess[0])):
                for cluster in final_clustering:
                    if guess[0][i][0] in cluster:
                        clust = list(cluster)

                        for word in guess[0][i]:
                            if word not in clust:
                                clust.append(word)
                        new_final_clustering.append(clust)

            for cluster in final_clustering:
                for clust in new_final_clustering:
                    if clust[0] in cluster:
                        for word in clust:
                            if word not in cluster:
                                cluster.append(word)
        #print(final_clustering)

        final_clustering.sort(key=len)
        #print(final_clustering)

        correct, wrong = self.get_clustering_result(final_clustering, reals)

        return correct, wrong

    #Needs final clustering to be sorted on length of clusters.
    #Does a lot of array handling to only have unique words in the clusters and compare the clusters with the real clusters
    #by taking the cluster with the most corrects of a cluster class and combines these two clusters, removing this option from the other clusters.
    # ONLY WORKS WITH 3 CLUSTERS!!!
    def get_clustering_result(self, final_clustering, reals):

        mediods = []
        for cluster in final_clustering:
            mediods.append(cluster[0])

        #print(mediods)

        all_unique_words = []
        for cluster in final_clustering:
            for word in cluster:
                if word not in all_unique_words:
                    all_unique_words.append(word)

        unique_final_clustering = []

        not_first = False

        #print(all_unique_words)
        for word in all_unique_words:
            if word in mediods: #First word should be first mediod
                if not_first:
                    unique_final_clustering.append(inner_unique_cluster)
                inner_unique_cluster = []
                not_first = True
            inner_unique_cluster.append(word) #Should not be a problem as first word should be first mediod
        unique_final_clustering.append(inner_unique_cluster)

        #print("testing")
        #print(unique_final_clustering)



        # Select first element of each cluster as element to compare against real cluster containing that element.
        upper_testset = []
        for test_cluster in reals:
            upper_test_cluster = []
            for word in test_cluster:
                upper_test_cluster.append(word.upper())
            upper_testset.append(upper_test_cluster)



        # Find accuracy of clusters by taking the one with most corrects and remove from each cluster list
        correct = []
        wrong = []
        i = 1
        for real_cluster in upper_testset:
            j = 1
            for final_cluster in unique_final_clustering:
                inner_correct = []
                inner_wrong = []
                for word in final_cluster:
                    if word in real_cluster:
                        inner_correct.append(word)
                    else:
                        inner_wrong.append(word)
                correct.append([inner_correct, i, j, len(inner_correct)])
                wrong.append([inner_wrong, i, j])
                j += 1
            i += 1

        #print(correct)
        correct.sort(key = lambda correct: correct[3], reverse = True)
        #print(correct)

        true_correct = []

        first_correct = list(correct[0])
        #print(first_correct)

        new_correct = []
        for i in range(0, len(correct)):
            if first_correct[1] != correct[i][1] and first_correct[2] != correct[i][2]:
                new_correct.append(correct[i])

        correct = list(new_correct)
        #print(correct)

        second_correct = list(correct[0])
        #print(second_correct)
        new_correct = []
        for i in range(0, len(correct)):
            if second_correct[1] != correct[i][1] and second_correct[2] != correct[i][2]:
                new_correct.append(correct[i])

        #print(new_correct)
        third_correct = list(new_correct[0])

        true_correct.append(first_correct)
        true_correct.append(second_correct)
        true_correct.append(third_correct)

        true_wrong = []
        for i in range(0, len(wrong)):
            if (first_correct[1] == wrong[i][1] and first_correct[2] == wrong[i][2]) or (second_correct[1] == wrong[i][1] and second_correct[2] == wrong[i][2]) or (third_correct[1] == wrong[i][1] and third_correct[2] == wrong[i][2]):
                true_wrong.append(wrong[i])


        #print("final")
        #print(unique_final_clustering)
        #print(true_correct)
        #print(true_wrong)

        final_correct = []
        final_wrong = []
        for correct in true_correct:
            for word in correct[0]:
                final_correct.append(word)

        for wrong in true_wrong:
            for word in wrong[0]:
                final_wrong.append(word)

        print(final_correct)
        print(final_wrong)

        return final_correct, final_wrong
        # remove duplicates across clusters by taking the words in the one with fewest and clear those from the others.



    def get_models_clustering_results(self, test_set, number_of_models):
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
                # If model knows less than 3 words we ignore it!
                try:
                    results.append(finished_model.get_model_clusters(test_set))
                except IndexError:
                    print("model knows to little")
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

#print(enseemble_test.get_human_similarities_results('wordsim353.tsv'))
#print(enseemble_test.get_model_similarities_results('wordsim353.tsv', 10))
#print(enseemble_test.naive_human_similarity_majority_vote('wordsim353.tsv', 10))
#print(enseemble_test.ignore_oov_human_similarity_majority_vote('wordsim353.tsv', 10))
#print(enseemble_test.weight_based_on_oov_human_similarity_majority_vote('wordsim353.tsv', 10))
#print(enseemble_test.weight_based_on_total_oov_ignore_oov_human_similarity_majority_vote('wordsim353.tsv', 10))


enseemble_test.naive_clustering_majority_vote('Nouns-verbs-adjectives.csv', 5)
enseemble_test.get_biggest_first_clustering_majority_vote('Nouns-verbs-adjectives.csv', 5)