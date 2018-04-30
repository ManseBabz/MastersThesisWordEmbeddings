import EnsamblePredictor as EP
import logging, os
from os import listdir
from os.path import isfile, join

def generate_array_of_all_trained_model_specification():
    name_array =[]
    not_wanted = 'npy'
    onlyfiles = [f for f in listdir(os.path.dirname(os.path.realpath(__file__))+"/LeaningAlgoImpl/Models") if isfile(join(os.path.dirname(os.path.realpath(__file__))+"/LeaningAlgoImpl/Models", f))]
    for file_index in range(0, len(onlyfiles)):
        if(not_wanted in onlyfiles[file_index]):
            continue
        else:
            name_array.append(onlyfiles[file_index])
    model_array = []
    for model in name_array:
        model_array.append(model.split(','))
    finished_array=[]
    for model in model_array:
        transformed_array_entry = [model[0]]
        placeholder_array = []
        for parameter_index in range(1, len(model)):
            placeholder_array.append(int(model[parameter_index]))
        transformed_array_entry.append(placeholder_array)
        finished_array.append(transformed_array_entry)
    return finished_array



def question_word_test():
    ensamble = EP.boot_strap_aggregator(generate_array_of_all_trained_model_specification())
    dir_path = os.path.dirname(os.path.realpath(__file__))+"/TestingSet/questions-words.txt"
    #ensamble.set_weights([1, 0.5])
    ensamble.accuracy(dir_path, predictor_method=0)

def word_sim_test():
    ensamble = EP.simple_ensamble(generate_array_of_all_trained_model_specification)
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/TestingSet/wordsim353.tsv"
    ensamble.evaluate_word_pairs(dir_path, similarity_model_type=0)

if __name__ == "__main__": question_word_test()