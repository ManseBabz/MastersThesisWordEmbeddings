import LeaningAlgoImpl
def train(algorithmName, trainset):
    algorithm = __import__(algorithmName, fromlist=[''])
    model = algorithm.train(trainset)
    return model

