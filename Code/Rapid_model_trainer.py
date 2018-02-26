import LeaningAlgoImpl.CBOW as CBOW
import LeaningAlgoImpl.Skip_Gram as Skip
def model_name_generator(model_type, model_params, training_articles):
    name = model_type
    for param in model_params:
        name += "," + str(param)
    name + '_Trained_on'+str(training_articles)+'articles'
    return name

def train_models():
    count = 0
    print('hmm')
    for hs in range (0, 2):
        for neg in range(5, 21, 4):
            for cbow_mean in range(0, 2):
                for iter in range(10, 51, 10):
                    for size in range (100, 201, 25):
                        for min_count in range(1, 10, 4):
                            CBOW = CBOW.CBOW(dev_mode=False)
                            CBOW.get_model(hs=hs, negative=neg, cbow_mean=cbow_mean, iter=iter, size=size, min_count=min_count,
                            max_vocab_size=1000000, workers=3, articles_to_learn=100000)
                            CBOW.save_model(model_name_generator('CBOW', [hs, neg, cbow_mean, iter, size, min_count], 100000))

                            Skip_gram = Skip.Skip_Gram(dev_mode=False)
                            Skip_gram.get_model(hs=hs, negative=neg, cbow_mean=cbow_mean, iter=iter, size=size,
                                           min_count=min_count,
                                           max_vocab_size=1000000, workers=3, articles_to_learn=100000)
                            Skip_gram.save_model(
                                model_name_generator('Skip_gram', [hs, neg, cbow_mean, iter, size, min_count], 100000))
                            count +=1
    print(count)
if __name__ == "__main__": train_models()