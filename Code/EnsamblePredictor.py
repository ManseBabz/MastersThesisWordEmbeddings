import LeaningAlgoImpl.CBOW as cbow
import predictor_setup as ps
import os, logging
"""
This module contains multiple simple ensamble methods
"""


"""
    simple_majority_vote_ensamble tests if multiple models get the same result, and
    chooses the best result from a majority vote i.e. the result which the most predictors
    got.
    If multiple equaly good results are found, the firs to raise above the others in count of
    ocurences in predictions wins (if all predictors find different results, the first
    predictors result is the "best result")
    This method returns the best result for a word, and the count for the word
"""
def result_unpacker(list_of_results):
    return_result = []
    for result_block in list_of_results:
        return_result = result_block
    return return_result

def simple_majority_vote_ensamble(leaner_list, word_list, top_n_words, wanted_printed=False, dev_mode=False):
    result = [] # List of results from the different predictors
    best_result = [None, 0] # Best result found

    """
        Setup of predictor classes
    """
    models = ps.setup(leaner_list, dev_mode)

    """
        Predict best word
    """
    for model in models:
        result.append(model.predict(word_list=word_list, nwords=top_n_words)) #Predict from set and add to result list

    """
        Majority vote for best word
    """
    print(result)
    result = result_unpacker(result)
    for res in result:
        if(result.count(res)> best_result[1]): #Check if next result has a better "score"
            best_result=[res, result.count(res)] #If better score, overwrite best result

    if(wanted_printed==True):
        print(word_list)
        print(best_result)
    return best_result


"""
    A better ensamble learner, where the results for
"""
def result_combinatrion_ensamble(params_list):
    print("not implemented yet")





"""
    The simple majority vote ensamble takes a 3 level array i.e. a tensor.
    The outer layer for defining what type of word-embedding to run. 
"""
if __name__ == "__main__": simple_majority_vote_ensamble([['Skip_Gram',[1,5,0,10,100,5,None,3]]], ['autonomous', 'individuals', 'mutual'], 4, wanted_printed=True, dev_mode=True)