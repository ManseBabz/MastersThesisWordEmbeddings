import numpy as np
import random
import os.path
from Our_FastText import utils
import keyboard
from os import listdir
from os.path import isfile, join
import LeaningAlgoImpl.Finished_Models as FM
import csv



def get_multiple_results(topn=1):
    questions = "questions-words.txt"
    name_array = []
    not_wanted = ['npy', 'Readme.md']
    onlyfiles = [f for f in listdir(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models") if
                 isfile(join(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models", f))]
    for file_index in range(0, len(onlyfiles)):
        if (not_wanted[0] in onlyfiles[file_index] or not_wanted[1] in onlyfiles[file_index]):
            continue
        else:
            name_array.append(onlyfiles[file_index])

    print(name_array)


    reals = get_expected_acc_results(questions)
    results = []
    for name in name_array:
        print(name)
        finished_model = FM.Finished_Models()
        finished_model.get_model(os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models/" + name)
        #models.append(finished_model)

        res = finished_model.get_acc_results(1, questions)
        correct = []
        wrong = []
        number_of_correct = 0
        number_of_wrong = 0
        for j in range(0, len(res)):
            if res[j] is not None:
                #print(res[j][0])

                predicted = res[j][0]
            else:
                predicted = res[j]
            expected = reals[j]
            if predicted == expected:
                correct_message = ("predicted: %s correct" % (predicted))
                correct.append(correct_message)
                number_of_correct += 1
            else:
                wrong_message = ("predicted: %s, should have been: %s" % (predicted, expected))
                wrong.append(wrong_message)
                number_of_wrong += 1
        results.append([name, len(correct), len(wrong)])


    print(reals)

    print(results)
    print(len(name_array))
    print(len(results))

    for res in results:
        print(res)
        if (os.path.isfile("individual_models_acc_english.csv")):
            f = open("individual_models_acc_english.csv", "a")
        else:
            f = open("individual_models_acc_english.csv", "w")

        np.savetxt(f, [res], delimiter=',', newline= "\n", fmt="%s")
        f.close()
    # print(results)
    return results


def get_model_similarities_results():
    questions = "wordsim353.tsv"
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
    results = []

    for name in name_array:
        print(name)
        finished_model = FM.Finished_Models()
        finished_model.get_model(
            os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models/" + name)
        pearson, spearman, oov = finished_model.human_similarity_test(questions)
        fixed_result = [name, 10, spearman[0], spearman[1], pearson[0], pearson[1]]
        print(fixed_result)
        results.append(fixed_result)

    print(results)

    for res in results:
        print(res)
        if (os.path.isfile("individual_models_human_similarity_english.csv")):
            f = open("individual_models_human_similarity_english.csv", "a")
        else:
            f = open("individual_models_human_similarity_english.csv", "w")

        np.savetxt(f, [res], delimiter=',', newline= "\n", fmt="%s")
        f.close()

    # print(results)
    return results


def get_expected_acc_results(questions):
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

def get_model_clustering_results():
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



    for test in ["Navneord-udsagnsord-tillægsord.csv", "Frugt-dyr-køretøjer.csv", "Hus-værktøj-kropsdele.csv"]:
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        test_set = dir_path + '/Code/TestingSet/' + test

        reals = []
        with open(test_set) as csvfile:
            test_reader = csv.reader(csvfile, delimiter=',')
            # initialize first cluster
            cluster = []
            for row in test_reader:
                if not row:
                    # Add new cluster
                    reals.append(cluster)
                    cluster = []
                else:
                    cluster.append(''.join(row))
            # add last cluster
            reals.append(cluster)

        print(reals)
        results = []
        for name in name_array:
            print(name)
            finished_model = FM.Finished_Models()
            finished_model.get_model(
                os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models/" + name)
            try:
                result = finished_model.clustering(reals)
                print(result)
                fixed_result = [name, result[1], result[3]]
                print(fixed_result)
                results.append(fixed_result)
            except IndexError:
                print("model knows to little")


        print(results)

        for res in results:
            print(res)
            if (os.path.isfile("individual_models_clustering_english" + test + ".csv")):
                f = open("individual_models_clustering_english" + test + ".csv", "a")
            else:
                f = open("individual_models_clustering_english" + test + ".csv", "w")

            np.savetxt(f, [res], delimiter=',', newline= "\n", fmt="%s")
            f.close()

    # print(results)
    return results


if __name__ == "__main__": get_multiple_results()