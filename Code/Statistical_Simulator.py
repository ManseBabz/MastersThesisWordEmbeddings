import Bootstrap_ensamble_word_embedding as BS
import numpy as np
import random
import os.path
from os.path import isfile, join
from os import listdir

"""Accuracy experiments"""

def generate_statistics(language, startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("Ensamble_test_results_"+language+".csv")):
                f = open("Ensamble_test_results_"+language+".csv", "a")
            else:
                f = open("Ensamble_test_results_"+language+".csv", "w")
            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "questions-words.txt"
            right, wrong = ensamble_model.accuracy(dir_path, number_of_models=i, language=language)
            res = [[i, topn, right, wrong]]
            results.append(res)
            print(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')
    print(results)
    return results

def acc_experiment1(language, startingpoint=150, endpoint=155, skips=5, iterations=1, topn=10):
    while True:
        start =random.randint(startingpoint, endpoint)
        generate_statistics(language=language, startingpoint=start,
                            endpoint=random.randint(start + 1, endpoint),
                            skips=random.randint(1, skips), iterations=random.randint(1, iterations),
                            topn=random.randint(1, topn))


def generate_statistics_with_weighted_majorityvote_ensamble(language, startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("weighted_majority_vote_Ensamble_test_results_"+language+".csv")):
                f = open("weighted_majority_vote_Ensamble_test_results_"+language+".csv", "a")
            else:
                f = open("weighted_majority_vote_Ensamble_test_results_"+language+".csv", "w")

            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "questions-words.txt"
            right, wrong = ensamble_model.accuracy(dir_path, number_of_models=i, predictor_method=3, language=language)
            res = [[i, topn, right, wrong]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')

    print(results)
    return results

def acc_experiment2(language, startingpoint=150, endpoint=155, skips=5, iterations=1, topn=10):
    while True:
        start =random.randint(startingpoint, endpoint)
        generate_statistics_with_weighted_majorityvote_ensamble(language=language, startingpoint=startingpoint,
                            endpoint=random.randint(start + 1, endpoint),
                            skips=random.randint(1, skips), iterations=random.randint(1, iterations),
                            topn=random.randint(1, topn))


def generate_statistics_with_weighted_tiebreaking_majorityvote_ensamble(language, startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("tie_breaking_weighted_majority_vote_"+language+".csv")):
                f = open("tie_breaking_weighted_majority_vote_"+language+".csv", "a")
            else:
                f = open("tie_breaking_weighted_majority_vote_"+language+".csv", "w")

            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "questions-words.txt"
            right, wrong = ensamble_model.accuracy(dir_path, number_of_models=i, predictor_method=4, language=language)
            res = [[i, topn, right, wrong]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')

    print(results)
    return results

def acc_experiment3(language, startingpoint=5, endpoint=160, skips=5, iterations=5, topn=10):
    while True:
        start =startingpoint
        generate_statistics_with_weighted_tiebreaking_majorityvote_ensamble(language=language, startingpoint=startingpoint,
                                                                endpoint=random.randint(start + 1, endpoint),
                                                                skips=random.randint(1, skips),
                                                                iterations=random.randint(1, iterations),
                                                                topn=random.randint(1, topn))


""" Human similarity experiments"""

def generate_statistics_naive_human_similarity(language, startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("naive_human_similarity_stats_"+language+".csv")):
                f = open("naive_human_similarity_stats_"+language+".csv", "a")
            else:
                f = open("naive_human_similarity_stats_"+language+".csv", "w")

            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "wordsim353.tsv"
            spearman_result, pearson_result = ensamble_model.evaluate_word_pairs(dir_path, number_of_models=i, similarity_model_type=0, language=language)
            print(spearman_result)
            res = [[i, topn, spearman_result[0], spearman_result[1], pearson_result[0], pearson_result[1]]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')

    print(results)
    return results

def humsim_experiment1(language, startingpoint=5, endpoint=160, skips=5, iterations=5, topn=10):
    while True:
        start =startingpoint
        generate_statistics_naive_human_similarity(language=language, startingpoint=startingpoint,
                                                                endpoint=random.randint(start + 1, endpoint),
                                                                skips=random.randint(1, skips),
                                                                iterations=random.randint(1, iterations))


def generate_statistics_ignore_oov_human_similarity(language, startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("ignore_oov_human_similarity_stats_"+language+".csv")):
                f = open("ignore_oov_human_similarity_stats_"+language+".csv", "a")
            else:
                f = open("ignore_oov_human_similarity_stats_"+language+".csv", "w")

            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "wordsim353.tsv"
            spearman_result, pearson_result = ensamble_model.evaluate_word_pairs(dir_path, number_of_models=i, similarity_model_type=1, language=language)
            res = [[i, topn, spearman_result[0], spearman_result[1], pearson_result[0], pearson_result[1]]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')

    print(results)
    return results

def humsim_experiment2(language, startingpoint=5, endpoint=160, skips=5, iterations=5, topn=10):
    while True:
        start =startingpoint
        generate_statistics_ignore_oov_human_similarity(language=language, startingpoint=startingpoint,
                                                                endpoint=random.randint(start + 1, endpoint),
                                                                skips=random.randint(1, skips),
                                                                iterations=random.randint(1, iterations))


def generate_statistics_weight_based_on_oov_human_similarity(language, startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("weight_based_on_oov_human_similarity_stats_"+language+".csv")):
                f = open("weight_based_on_oov_human_similarity_stats_"+language+".csv", "a")
            else:
                f = open("weight_based_on_oov_human_similarity_stats_"+language+".csv", "w")

            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "wordsim353.tsv"
            spearman_result, pearson_result = ensamble_model.evaluate_word_pairs(dir_path, number_of_models=i, similarity_model_type=2,language=language)
            res = [[i, topn, spearman_result[0], spearman_result[1], pearson_result[0], pearson_result[1]]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')

    print(results)
    return results

def humsim_experiment3(language, startingpoint=5, endpoint=160, skips=5, iterations=5, topn=10):
    while True:
        start =startingpoint
        generate_statistics_weight_based_on_oov_human_similarity(language=language, startingpoint=startingpoint,
                                                                endpoint=random.randint(start + 1, endpoint),
                                                                skips=random.randint(1, skips),
                                                                iterations=random.randint(1, iterations))

def generate_statistics_weight_based_on_total_oov_ignore_oov_human_similarity(language, startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("weight_based_on_total_oov_ignore_oov_human_similarity_stats_"+language+".csv")):
                f = open("weight_based_on_total_oov_ignore_oov_human_similarity_stats_"+language+".csv", "a")
            else:
                f = open("weight_based_on_total_oov_ignore_oov_human_similarity_stats_"+language+".csv", "w")

            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "wordsim353.tsv"
            spearman_result, pearson_result = ensamble_model.evaluate_word_pairs(dir_path, number_of_models=i, similarity_model_type=3, language=language)
            res = [[i, topn, spearman_result[0], spearman_result[1], pearson_result[0], pearson_result[1]]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')

    print(results)
    return results

def humsim_experiment4(language, startingpoint=5, endpoint=160, skips=5, iterations=5, topn=10):
    while True:
        start =startingpoint
        generate_statistics_weight_based_on_total_oov_ignore_oov_human_similarity(language=language, startingpoint=155,
                                                                endpoint=random.randint(start + 1, endpoint),
                                                                skips=random.randint(1, skips),
                                                                iterations=random.randint(1, iterations))

def all_hum_sim(language, startingpoint=5, endpoint=100, skips=5, iterations=5):
        """generate_statistics_weight_based_on_total_oov_ignore_oov_human_similarity(language=language, startingpoint=160,
                                                                endpoint=endpoint,
                                                                skips=skips,
                                                                iterations=iterations)
        generate_statistics_weight_based_on_oov_human_similarity(language=language, startingpoint=195,
                                                                 endpoint=196,
                                                                 skips=skips,
                                                                 iterations=iterations)"""
        generate_statistics_ignore_oov_human_similarity(language=language, startingpoint=195,
                                                        endpoint=196,
                                                        skips=skips,
                                                        iterations=iterations)
        generate_statistics_naive_human_similarity(language=language, startingpoint=195,
                                                   endpoint=196,
                                                   skips=skips,
                                                   iterations=iterations)



def model_counter():
    name_array = []
    not_wanted = ['npy', 'Readme.md']
    onlyfiles = [f for f in listdir(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models") if
                 isfile(join(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models", f))]
    for file_index in range(0, len(onlyfiles)):
        if (not_wanted[0] in onlyfiles[file_index] or not_wanted[1] in onlyfiles[file_index]):
            continue
        else:
            name_array.append(onlyfiles[file_index])
    #print(len(name_array))
    return len(name_array)

def oov_test(language, startingpoint, endpoint, skips, iterations):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("oov_test_"+language+".csv")):
                f = open("oov_test_"+language+".csv", "a")
            else:
                f = open("oov_test_"+language+".csv", "w")

            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "questions-words.txt"
            oov = ensamble_model.oov_test(questions=dir_path, number_of_models=i, language=language)
            res = [[i, oov]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')

def oov_experiment(language, startingpoint=5, endpoint=160, skips=5, iterations=5):
    print("oov_test initiated")
    while True:
        oov_test(language=language, startingpoint=startingpoint,
                 endpoint=random.randint(startingpoint + 1, endpoint),
                 skips=random.randint(1, skips),
                 iterations=random.randint(1, iterations))


""" Clustering experiments"""
def generate_clusters_naive(language, test_set, startingpoint = 5, endpoint = 50, skips = 5, iterations = 5):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("naive_clustering_"+language+"_" + test_set + + ".csv")):
                f = open("naive_clustering_"+language+"_" + test_set + ".csv", "a")
            else:
                f = open("naive_clustering_"+language+"_" + test_set + ".csv", "w")

            ensamble_model = BS.boot_strap_aggregator()
            correct, wrong, length_of_testset = ensamble_model.ensemble_clusters(test_set, number_of_models=i, clustering_type=0, language=language)
            res = [[i, len(correct), len(wrong), length_of_testset]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')

    print(results)
    return results

def cluster_experiment_1(language, startingpoint=5, endpoint=160, skips=5, iterations=5):
    test_sets = ["Navneord-udsagnsord-tillægsord.csv", "Frugt-dyr-køretøjer.csv", "Hus-værktøj-kropsdele.csv"]
    while True:
        start = startingpoint
        generate_clusters_naive(language, startingpoint=startingpoint,
                                                                endpoint=random.randint(start + 1, endpoint),
                                                                skips=random.randint(1, skips),
                                                                iterations=random.randint(1, iterations),
                                                                test_set=test_sets[random.randint(0, len(test_sets))])

def generate_clusters_biggest_first(language, test_set, startingpoint = 5, endpoint = 50, skips = 5, iterations = 5):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("biggest_first_clustering_"+language+"_" + test_set + ".csv")):
                f = open("biggest_first_clustering_"+language+"_" + test_set + ".csv", "a")
            else:
                f = open("biggest_first_clustering_"+language+"_" + test_set + ".csv", "w")

            ensamble_model = BS.boot_strap_aggregator()
            correct, wrong, length_of_testset = ensamble_model.ensemble_clusters(test_set, number_of_models=i, clustering_type=1, language=language)
            res = [[i, len(correct), len(wrong), length_of_testset]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')

    print(results)
    return results

def cluster_experiment_2(language, startingpoint=5, endpoint=160, skips=5, iterations=5):
    test_sets = ["Navneord-udsagnsord-tillægsord.csv", "Frugt-dyr-køretøjer.csv", "Hus-værktøj-kropsdele.csv"]
    while True:
        start = startingpoint
        generate_clusters_biggest_first(language, startingpoint=startingpoint,
                                                                endpoint=random.randint(start + 1, endpoint),
                                                                skips=random.randint(1, skips),
                                                                iterations=random.randint(1, iterations),
                                                                test_set=test_sets[random.randint(0, len(test_sets))])






if __name__ == "__main__": oov_experiment(startingpoint=33, language='English')