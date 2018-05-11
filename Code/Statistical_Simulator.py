import EnsamblePredictor as EP
import numpy as np
import random
import os.path
import keyboard

def generate_statistics(startingpoint = 5, endpoint = 50, skips = 5, iterations = 5, topn=10):
    results = []
    if(os.path.isfile("Ensamble_test_results.csv") ):
        f = open("Ensamble_test_results.csv", "a")
    else:
        f = open("Ensamble_test_results.csv", "w")
    for i in range(startingpoint, endpoint, skips):
        for j in range(0, iterations):
            print(j)
            ensamble_model = EP.boot_strap_aggregator()
            dir_path = "questions-words.txt"
            right, wrong = ensamble_model.accuracy(dir_path, number_of_models=i)
            res = [[i, topn, right, wrong]]
            print(res)
            results.append(res)
            if (os.path.isfile("Ensamble_test_results.csv")):
                f = open("Ensamble_test_results.csv", "a")
            else:
                f = open("Ensamble_test_results.csv", "w")
            np.savetxt(f, res, delimiter=',')
            f.close()
    print(results)

    return results

def experiment(startingpoint=5, endpoint=400, skips=50, iterations=5, topn=10):
    while True:
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed(' '):  # if key 'q' is pressed

                print('finishing')
                break  # finishing the loop
            else:
                generate_statistics(startingpoint=random.randint(1, startingpoint),
                                    endpoint=random.randint(startingpoint + 1, endpoint),
                                    skips=random.randint(1, skips), iterations=random.randint(1, iterations),
                                    topn=random.randint(1, topn))
        except:
            break



if __name__ == "__main__": experiment()