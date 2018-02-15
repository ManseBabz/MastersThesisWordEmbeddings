import LearningAlgorithm
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
"""
def simple_majority_vote_ensamble(ListOfAlgorithms, ListOfPreTrainedModels):
    Trainset = []
    algorithms = []
    models = []
    results =[]
    # Initialize the different
    for names in ListOfAlgorithms:
        algorithms.append(LearningAlgorithm(names, Trainset))
    # Create the different submodels used for the ensamble learning
    for alg in algorithms:
        models.append(alg.train(Trainset))
        print("Training new algorithm")
    # Import previously created models, in order to save time by not
    # training models already trained
    for givenModel in ListOfPreTrainedModels:
        models.append(givenModel)
    for predictionModel in models:
        results.append(predictionModel.predict())
    # Predict using ensamble learning - simple majority vote
    bestCount=0
    for result in results:
        if(bestCount < results.count(result)):
            bestresult =result
    return bestresult

