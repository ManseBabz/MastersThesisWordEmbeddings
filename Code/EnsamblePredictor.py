import LeaningAlgoImpl.CBOW as cbow
import predictor_setup as ps
import os, logging, operator
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
def simple_majority_vote_ensamble(leaner_list, word_list, top_n_words, training_articles=1000, wanted_printed=False, dev_mode=False):
    result = [] # List of results from the different predictors
    best_result = [None, 0] # Best result found

    #Setup of predictor classes
    models = ps.setup(leaner_list, dev_mode=dev_mode, training_articles=1000)

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
def result_combinatrion_ensamble(leaner_list, word_list, top_n_words, training_articles=1000, wanted_printed=False, dev_mode=False):
    result = []  # List of results from the different predictors
    best_result = [None, 0]  # Best result found

    #Setup of predictor classes
    models = ps.setup(leaner_list, dev_mode=dev_mode, training_articles=1000)

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
"""
def boot_strap_aggregator_predictor():
    print("Not implemented yet")


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
if __name__ == "__main__": result_combinatrion_ensamble([
    ['CBOW',[1,5,0,10,100,500,None,3]],
    ['CBOW', [2, 50, 0, 10, 100, 500, None, 3]],
    ['CBOW', [5, 5, 5, 10, 100, 5000, None, 3]],
    ['CBOW', [1, 500, 0, 10, 100, 5000, None, 3]],
    ['CBOW', [1, 5, 10, 50, 100, 5, None, 3]],
    ['Skip_Gram',[1,5,0,10,100,5,None,3]],
    ['Skip_Gram', [1, 5, 0, 10, 100, 500, None, 3]],
    ['Skip_Gram', [2, 50, 0, 10, 100, 500, None, 3]],
    ['Skip_Gram', [5, 5, 5, 10, 100, 5000, None, 3]],
    ['Skip_Gram', [1, 500, 0, 10, 100, 5000, None, 3]],
    ['Fast_Text', [1, 5, 10, 50, 1000, 50, None, 3]]], ['he', 'she', 'his'], 4, training_articles=1000000, wanted_printed=True, dev_mode=True)