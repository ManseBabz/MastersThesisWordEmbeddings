import LearningAlgorithm
"""
This module will train a Stacking model, which can be used for word embeddings
"""




def trainstackinsodeltrainer(models):
    #train on the combined set of submodels to make a combined model
    print("training Stacking model")
    """
        Insert code for density estimation
    """
    finishedModel =[]
    return finishedModel

def main(ListOfAlgorithms, ListOfPreTrainedModels):
    Trainset = []
    algorithms = []
    models = []

    #Initialize the different
    for names in ListOfAlgorithms:
        algorithms.append(LearningAlgorithm(names, Trainset))

    #Create the different submodels used for the ensamble learning
    for alg in algorithms:
        models.append(alg.train(Trainset))
        print("Training new algorithm")

    #Import previously created models, in order to save time by not
    # training models already trained
    for givenModel in ListOfPreTrainedModels:
        models.append(givenModel)

    #Train the stackmodel
    finishedModel = trainstackinsodeltrainer(models)
    print("Training is finished")
    return finishedModel

if __name__ == "__main__": main()