import Bootstrap_ensamble_word_embedding as BS
import numpy as np
import random
import os.path
import keyboard

def generate_statistics(startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            if (os.path.isfile("Ensamble_test_results.csv")):
                f = open("Ensamble_test_results.csv", "a")
            else:
                f = open("Ensamble_test_results.csv", "w")
            print(j)
            ensamble_model = BS.boot_strap_aggregator()
            dir_path = "questions-words.txt"
            right, wrong = ensamble_model.accuracy(dir_path, number_of_models=i)
            res = [[i, topn, right, wrong]]
            print(res)
            results.append(res)
            np.savetxt(f, res, delimiter=',')
            f.close()
            print('iteration finished')
    print(results)
    return results

def experiment1(startingpoint=5, endpoint=30, skips=5, iterations=5, topn=10):
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
            print(j)
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

def experiment2(startingpoint=150, endpoint=155, skips=5, iterations=1, topn=10):
    while True:
        start =random.randint(startingpoint, endpoint)
        generate_statistics_with_weighted_majorityvote_ensamble(startingpoint=startingpoint,
                            endpoint=random.randint(start + 1, endpoint),
                            skips=random.randint(1, skips), iterations=random.randint(1, iterations),
                            topn=random.randint(1, topn))




if __name__ == "__main__": experiment2()