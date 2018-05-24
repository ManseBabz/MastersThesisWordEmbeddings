import Bootstrap_ensamble_word_embedding as BS
import numpy as np
import random
import os.path
import keyboard

"""Accuracy experiments"""

def generate_statistics(startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("Ensamble_test_results.csv")):
                f = open("Ensamble_test_results.csv", "a")
            else:
                f = open("Ensamble_test_results.csv", "w")
            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "questions-words.txt"
            right, wrong = ensamble_model.accuracy(dir_path, number_of_models=i)
            res = [[i, topn, right, wrong]]
            results.append(res)
            print(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')
    print(results)
    return results

def acc_experiment1(startingpoint=150, endpoint=155, skips=5, iterations=1, topn=10):
    while True:
        start =random.randint(startingpoint, endpoint)
        generate_statistics(startingpoint=start,
                            endpoint=random.randint(start + 1, endpoint),
                            skips=random.randint(1, skips), iterations=random.randint(1, iterations),
                            topn=random.randint(1, topn))


def generate_statistics_with_weighted_majorityvote_ensamble(startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("weighted_majority_vote_Ensamble_test_results.csv")):
                f = open("weighted_majority_vote_Ensamble_test_results.csv", "a")
            else:
                f = open("weighted_majority_vote_Ensamble_test_results.csv", "w")

            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "questions-words.txt"
            right, wrong = ensamble_model.accuracy(dir_path, number_of_models=i, predictor_method=3)
            res = [[i, topn, right, wrong]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')

    print(results)
    return results

def acc_experiment2(startingpoint=150, endpoint=155, skips=5, iterations=1, topn=10):
    while True:
        start =random.randint(startingpoint, endpoint)
        generate_statistics_with_weighted_majorityvote_ensamble(startingpoint=startingpoint,
                            endpoint=random.randint(start + 1, endpoint),
                            skips=random.randint(1, skips), iterations=random.randint(1, iterations),
                            topn=random.randint(1, topn))


def generate_statistics_with_weighted_tiebreaking_majorityvote_ensamble(startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("tie_breaking_weighted_majority_vote.csv")):
                f = open("tie_breaking_weighted_majority_vote.csv", "a")
            else:
                f = open("tie_breaking_weighted_majority_vote.csv", "w")

            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "questions-words.txt"
            right, wrong = ensamble_model.accuracy(dir_path, number_of_models=i, predictor_method=4)
            res = [[i, topn, right, wrong]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')

    print(results)
    return results

def acc_experiment3(startingpoint=5, endpoint=160, skips=5, iterations=5, topn=10):
    while True:
        start =startingpoint
        generate_statistics_with_weighted_tiebreaking_majorityvote_ensamble(startingpoint=startingpoint,
                                                                endpoint=random.randint(start + 1, endpoint),
                                                                skips=random.randint(1, skips),
                                                                iterations=random.randint(1, iterations),
                                                                topn=random.randint(1, topn))


""" Human similarity experiments"""

def generate_statistics_naive_human_similarity(startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("naive_human_similarity_stats.csv")):
                f = open("naive_human_similarity_stats.csv", "a")
            else:
                f = open("naive_human_similarity_stats.csv", "w")

            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "wordsim353.tsv"
            spearman_result, pearson_result = ensamble_model.evaluate_word_pairs(dir_path, number_of_models=i, similarity_model_type=0)
            res = [[i, topn, spearman_result, pearson_result]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')

    print(results)
    return results

def humsim_experiment1(startingpoint=5, endpoint=160, skips=5, iterations=5, topn=10):
    while True:
        start =startingpoint
        generate_statistics_naive_human_similarity(startingpoint=startingpoint,
                                                                endpoint=random.randint(start + 1, endpoint),
                                                                skips=random.randint(1, skips),
                                                                iterations=random.randint(1, iterations),
                                                                topn=random.randint(1, topn))


def generate_statistics_ignore_oov_human_similarity(startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("ignore_oov_human_similarity_stats.csv")):
                f = open("ignore_oov_human_similarity_stats.csv", "a")
            else:
                f = open("ignore_oov_human_similarity_stats.csv", "w")

            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "wordsim353.tsv"
            spearman_result, pearson_result = ensamble_model.evaluate_word_pairs(dir_path, number_of_models=i, similarity_model_type=1)
            res = [[i, topn, spearman_result, pearson_result]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')

    print(results)
    return results

def humsim_experiment2(startingpoint=5, endpoint=160, skips=5, iterations=5, topn=10):
    while True:
        start =startingpoint
        generate_statistics_ignore_oov_human_similarity(startingpoint=startingpoint,
                                                                endpoint=random.randint(start + 1, endpoint),
                                                                skips=random.randint(1, skips),
                                                                iterations=random.randint(1, iterations),
                                                                topn=random.randint(1, topn))


def generate_statistics_weight_based_on_oov_human_similarity(startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("weight_based_on_oov_human_similarity_stats.csv")):
                f = open("weight_based_on_oov_human_similarity_stats.csv", "a")
            else:
                f = open("weight_based_on_oov_human_similarity_stats.csv", "w")

            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "wordsim353.tsv"
            spearman_result, pearson_result = ensamble_model.evaluate_word_pairs(dir_path, number_of_models=i, similarity_model_type=2)
            res = [[i, topn, spearman_result, pearson_result]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')

    print(results)
    return results

def humsim_experiment3(startingpoint=5, endpoint=160, skips=5, iterations=5, topn=10):
    while True:
        start =startingpoint
        generate_statistics_weight_based_on_oov_human_similarity(startingpoint=startingpoint,
                                                                endpoint=random.randint(start + 1, endpoint),
                                                                skips=random.randint(1, skips),
                                                                iterations=random.randint(1, iterations),
                                                                topn=random.randint(1, topn))

def generate_statistics_weight_based_on_total_oov_ignore_oov_human_similarity(startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("weight_based_on_total_oov_ignore_oov_human_similarity_stats.csv")):
                f = open("weight_based_on_total_oov_ignore_oov_human_similarity_stats.csv", "a")
            else:
                f = open("weight_based_on_total_oov_ignore_oov_human_similarity_stats.csv", "w")

            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "wordsim353.tsv"
            spearman_result, pearson_result = ensamble_model.evaluate_word_pairs(dir_path, number_of_models=i, similarity_model_type=3)
            res = [[i, topn, spearman_result, pearson_result]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')

    print(results)
    return results

def humsim_experiment4(startingpoint=5, endpoint=160, skips=5, iterations=5, topn=10):
    while True:
        start =startingpoint
        generate_statistics_weight_based_on_total_oov_ignore_oov_human_similarity(startingpoint=startingpoint,
                                                                endpoint=random.randint(start + 1, endpoint),
                                                                skips=random.randint(1, skips),
                                                                iterations=random.randint(1, iterations),
                                                                topn=random.randint(1, topn))


if __name__ == "__main__": acc_experiment3(startingpoint=25)