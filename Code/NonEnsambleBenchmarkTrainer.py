import LeaningAlgoImpl.CBOW as CB
import LeaningAlgoImpl.Skip_Gram as Skip
import LeaningAlgoImpl.Fast_Text as FT
import time
import os

def model_name_generator(model_type, model_params, training_articles):
    name = model_type
    for param in model_params:
        name += "," + str(param)
    name + '_Trained_on'+str(training_articles)+'articles'
    return name


def model_exists_checker(model_name, dev_mode):
    if dev_mode:
        model_path = os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/DevModels/" + model_name
    else:
        model_path = os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models/"+model_name
    if(os.path.exists(model_path)):
        return True
    else:
        return False

def train_models():
    CBOW = CB.CBOW(dev_mode=False)
    model_name = model_name_generator('CBOW', [1, 10, 1, 10, 1000, 5, 10000000], 1000000)
    if model_exists_checker(model_name, dev_mode=False):
        print("Model already exists")
    else:
        start_time = time.time()
        CBOW.get_model(hs=1, negative=10, cbow_mean=1, iter=10, size=1000, min_count=5,
                       max_vocab_size=10000000, workers=3, articles_to_learn=1000000, randomTrain=True)
        CBOW.save_model(model_name)
        print("CBOW train-time " + str(time.time()-start_time))

    # train the SkipGram model
    Skip_gram = Skip.Skip_Gram(dev_mode=False)
    model_name = model_name_generator('Skip_gram', [1, 10, 0, 10, 1000, 5, 10000000], 1000000)
    if model_exists_checker(model_name, dev_mode=False):
        print("Model already exists")
    else:
        start_time = time.time()
        Skip_gram.get_model(hs=1, negative=10, cbow_mean=0, iter=10, size=1000, min_count=5,
                       max_vocab_size=10000000, workers=3, articles_to_learn=1000000, randomTrain=True)
        Skip_gram.save_model(model_name)
        print("Skip-Gram train-time " + str(time.time() - start_time))

    Fast_Text = FT.Fast_Text(dev_mode=False)
    model_name = model_name_generator('Fast_Text', [1, 10, 1, 10, 1000, 5, 10000000], 1000000)
    if model_exists_checker(model_name, dev_mode=False):
        print("Model already exists")
    else:
        start_time = time.time()
        Fast_Text.get_model(hs=1, negative=10, cbow_mean=1, iter=10, size=1000, min_count=5,
                       max_vocab_size=10000000, workers=3, articles_to_learn=1000000, randomTrain=True)
        Fast_Text.save_model(model_name)
        print("FastText train-time " + str(time.time() - start_time))

if __name__ == "__main__": train_models()