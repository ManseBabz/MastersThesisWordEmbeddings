import LeaningAlgoImpl.CBOW as cbow
import StackingModelTrainer as l
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
def simple_majority_vote_ensamble(leaner_list, word_list, top_n_words):
    models = [] # List of model-classes
    result = [] # List of results from the different predictors
    best_result = [None, 0] # Best result found

    """
        Setup of predictor classes
    """
    for t in leaner_list:
        if(t[0] == 'CBOW'):
            i=t[1]
            mod = cbow.CBOW()  # Initialize model
            mod.get_model(hs =i[0], negative= i[1], cbow_mean=i[2], iter= i[3], size=i[4], min_count=i[5], max_vocab_size=i[6], workers=i[7]) #Train model
            models.append(mod) #Add model class to list of trained model classes

    """
        Predict best word
    """
    for model in models:
        print(model.predict(word_list=word_list, nwords=top_n_words))
        result.append(model.predict(word_list=word_list, nwords=top_n_words)) #Predict from set and add to result list
    print(result)
    """
        Majority vote for best word
    """
    for res in result:
        if(result.count(res)> best_result[1]): #Check if next result has a better "score"
            best_result=[res, result.count(res)] #If better score, overwrite best result
    print(best_result)
    return  best_result


"""
    A better ensamble learner, where the results for
"""
def result_combinatrion_ensamble(params_list):
    print("not implemented yet")





"""
    The simple majority vote ensamble takes a 3 level array i.e. a tensor.
    The outer layer for defining what type of word-embedding to run. 
"""
if __name__ == "__main__": simple_majority_vote_ensamble([['CBOW',[1,5,0,10,100,5,None,3]]], ['What', 'is'], 4)