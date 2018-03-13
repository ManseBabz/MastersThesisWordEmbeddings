import predictor_setup as ps
import os, operator
import numpy as np
"""
This module contains multiple ensamble methods
"""

####################################################################################################################################
#################################### Helper methods ################################################################################
####################################################################################################################################
def result_unpacker(list_of_results):
    return_result = []
    print(list_of_results)
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

def contains(list_of_results, keyword):
    truth_value = False
    for string in list_of_results:
        if(string == keyword):
            truth_value =True
    return truth_value

def update_prob(list_of_results, key_word, prob_value):
    for entry in list_of_results:
        if(entry[0]==key_word):
            entry[1] += prob_value
    return list_of_results

####################################################################################################################################
#################################### Ensamble Methods ##############################################################################
####################################################################################################################################
"""
    simple_majority_vote_ensamble tests if multiple models get the same result, and
    chooses the best result from a majority vote i.e. the result which the most predictors
    got.
    If multiple equaly good results are found, the firs to raise above the others in count of
    ocurences in predictions wins (if all predictors find different results, the first
    predictors result is the "best result")
    This method returns the best result for a word, and the count for the word
"""
def simple_majority_vote_ensamble(leaner_list, word_list, top_n_words, training_articles=1000,
                                  wanted_printed=False, dev_mode=False):
    result = [] # List of results from the different predictors
    best_result = [None, 0] # Best result found

    #Setup of predictor classes
    models = ps.setup(leaner_list, dev_mode=dev_mode, training_articles=training_articles)

    #Predict best word
    for model in models:
        result.append(model.predict(word_list=word_list, nwords=top_n_words)) #Predict from set and add to result list

    #Majority vote for best word
    result = result_unpacker(result)
    result = remove_probability_from_result_list(result)
    for res in result:
        if(result.count(res)> best_result[1]): #Check if next result has a better "score"
            best_result=[res, result.count(res)] #If better score, overwrite best result

    if(wanted_printed==True):
        print(word_list)
        print(best_result)
    return best_result


"""
    Ensamble learner which naively finds the most probable result by adding the probability from each model for a word
"""
def most_probable_ensamble(leaner_list, word_list, top_n_words, training_articles=1000, wanted_printed=False, dev_mode=False):
    result = []  # List of results from the different predictors
    best_result = [None, 0]  # Best result found

    #Setup of predictor classes
    models = ps.setup(leaner_list, dev_mode=dev_mode, training_articles=training_articles)

    #Predict best word
    for model in models:
        result.append(model.predict(word_list=word_list, nwords=top_n_words))  # Predict from set and add to result list

    #Naivly find the most probable result
    result = result_unpacker(result)
    most_probable_result_storing = []
    for res in result:
        if(contains(most_probable_result_storing, res[0])):
            update_prob(most_probable_result_storing, res[0], res[1])
        else:
            most_probable_result_storing.append(res)
    most_probable_result_storing.sort(key=operator.itemgetter(1), reverse=True)
    best_result=most_probable_result_storing[0]

    if (wanted_printed == True):
        print(word_list)
        print(most_probable_result_storing)
        print(best_result)
    return best_result

"""
    Bootstrap aggregation model
    
    This model dosn't have a development mode due to the random sampeling nature of the ensamble method
"""
def boot_strap_aggregator_predictor(leaner_list, positive_word_list, negative_word_list, top_n_words,
                                    training_articles=1000, wanted_printed=False):
    result = [] # List of results from the different predictors
    best_result = [None, 0] # Best result found

    #Setup of predictor classes
    models = ps.setup(leaner_list, dev_mode=False, training_articles=training_articles, randomTrain=True)

    #Predict best word
    for model in models:
        result.append(model.predict(positive_word_list=positive_word_list, negative_word_list=negative_word_list)) #Predict from set and add to result list

    #Majority vote for best word
    result = result_unpacker(result)
    result = remove_probability_from_result_list(result)
    for res in result:
        if(result.count(res)> best_result[1]): #Check if next result has a better "score"
            best_result=[res, result.count(res)] #If better score, overwrite best result

    if(wanted_printed==True):
        print(best_result)
    return best_result

def boot_strap_aggregator_predictor_with_weights(leaner_list, weight_list, positive_word_list, negative_word_list, top_n_words,
                                                 training_articles=1000, wanted_printed=False):
    result = []  # List of results from the different predictors
    best_result = [None, 0]  # Best result found

    # Setup of predictor classes
    models = ps.setup(leaner_list, dev_mode=False, training_articles=training_articles, randomTrain=True)

    # Predict best word and calculate weightet probability for this word
    for i in range(0, len(models)):
        res = models[i].predict(positive_word_list=positive_word_list,
                                         negative_word_list=negative_word_list)  # Predict from set and add to result list
        for part_res in res:
            temp_res = []
            temp_res.append(part_res[0])
            temp_res.append(part_res[1]*weight_list[i])
            result.append(temp_res)

    # Combine probability for results
    most_probable_result_storing = []
    for res in result:
        if (contains(most_probable_result_storing, res[0])):
            update_prob(most_probable_result_storing, res[0], res[1])
        else:
            most_probable_result_storing.append(res)
    most_probable_result_storing.sort(key=operator.itemgetter(1), reverse=True)

    # Pick best result
    if(most_probable_result_storing != []):
        if (wanted_printed == True):
            print(most_probable_result_storing[0])
        return most_probable_result_storing[0]
    else:
        print("No result")
        return []


def stacking_model_trainer(leaner_list, weight_file_name):
    savepath = os.path.dirname(os.path.realpath(__file__))+"/LeaningAlgoImpl/Weight_models/"+weight_file_name
    weights = []
    # TODO - Make a way to train the model
    learned_result_to_file = [leaner_list, weights]
    np.save(savepath, learned_result_to_file)
    return learned_result_to_file[1]

def stacking_model_predictor(leaner_list, positive_word_list, negative_word_list, training_articles=1000, wanted_printed=False,
                             weight_file_param=None, weight_file_name="Simple_weight_file"):
    if(weight_file_param == None):
        weights = stacking_model_trainer(leaner_list, weight_file_name="Simple_weight_file")
        print("save using "+weight_file_name)
    else:
        weights=weight_file_param
    result = boot_strap_aggregator_predictor_with_weights(leaner_list=leaner_list, weight_list=weights, positive_word_list=positive_word_list,
                                                          negative_word_list=negative_word_list, training_articles=training_articles, wanted_printed=wanted_printed)
    return result

def loaded_stacking_model(stacking_model_file_path, positive_word_list, negative_word_list):
    models_and_weights = np.load(stacking_model_file_path)
    model_list=models_and_weights[0]
    weight_list=models_and_weights[1]
    res = stacking_model_predictor(leaner_list=model_list, positive_word_list=positive_word_list, negative_word_list=negative_word_list,
                                   training_articles=1000, wanted_printed=False, weight_file_param=weight_list)
    return res
####################################################################################################################################
#################################### Main method for testing pourpuse ##############################################################
####################################################################################################################################

"""
    The simple majority vote ensamble takes a 3 level array i.e. a tensor.
    The outer layer for defining what type of word-embedding to run. 
    
    Array arguments
    0 - hs=hs, #1 for hierarchical softmax and 0 and non-zero in negative argument then negative sampling is used.
    1 - negative=negative, #0 for no negative sampling and above specifies how many noise words should be drawn. (Usually 5-20 is good).
    2 - cbow_mean=cbow_mean, #0 for sum of context vectors, 1 for mean of context vectors. Only used on CBOW.
    3 - iter=iter, #number of epochs.
    4 - size=size, #feature vector dimensionality
    5 - min_count=min_count, #minimum frequency of words required
    6 - max_vocab_size=max_vocab_size, #How much RAM is allowed, 10 million words needs approx 1GB RAM. None = infinite RAM
    7 - workers=workers, #How many threads are started for training.
    
"""
if __name__ == "__main__": boot_strap_aggregator_predictor_with_weights([
    ['Special_Fast_Text',[1,5,0,10,100,500,None,3]]],
    [1],
    ['he', 'she'],
    ['what'],
    4,
    training_articles=1000,
    wanted_printed=True)