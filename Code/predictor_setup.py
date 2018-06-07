import LeaningAlgoImpl.CBOW as CBOW
import LeaningAlgoImpl.Skip_Gram as Skip
import LeaningAlgoImpl.Fast_Text as Fast_Text
import LeaningAlgoImpl.own_fast_text_test as Special
import os


def model_name_generator(model_type, model_params, training_articles):
    name = model_type
    for param in model_params:
        name += "," + str(param)
    name + '_Trained_on'+str(training_articles)+'articles'
    return name

def model_exists_checker(model_name, dev_mode, language):
    if dev_mode:
        model_path = os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/DevModels/"+language+'/' + model_name
    else:
        model_path = os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models/"+language+'/'+model_name
    if(os.path.exists(model_path)):
        return True
    else:
        return False

def setup(leaner_list, language, dev_mode, training_articles=1000, randomTrain=False):
    models = []
    for t in leaner_list:
        if (t[0] == 'CBOW'):
            mod = CBOW.CBOW(dev_mode=dev_mode)  # Initialize CBOW model
            model_name = model_name_generator(t[0], t[1], training_articles)
            if model_exists_checker(model_name, dev_mode=dev_mode, language=language):
                mod.load_model(model_name, language)#Not finished
                models.append(mod)
            else:
                i = t[1]
                mod.get_model(language=language, hs=i[0], negative=i[1], cbow_mean=i[2], iter=i[3], size=i[4], min_count=i[5],
                          max_vocab_size=i[6], workers=i[7], articles_to_learn=training_articles, randomTrain=randomTrain)  # Train CBOW model
                mod.finished_training()
                mod.save_finished_model(model_name, language)
                models.append(mod)  # Add model class to list of trained model classes
        elif (t[0] == 'Skip_Gram'):
            mod = Skip.Skip_Gram(dev_mode=dev_mode)  # Initialize CBOW model
            model_name = model_name_generator(t[0], t[1], training_articles)
            if model_exists_checker(model_name, dev_mode=dev_mode, language=language):
                mod.load_model(model_name, language)#Not finished
                models.append(mod)
            else:
                i = t[1]
                mod.get_model(language=language, hs=i[0], negative=i[1], cbow_mean=i[2], iter=i[3], size=i[4], min_count=i[5],
                              max_vocab_size=i[6], workers=i[7], articles_to_learn=training_articles, randomTrain=randomTrain)  # Train model
                mod.finished_training()
                mod.save_finished_model(model_name, language)
                models.append(mod)  # Add model class to list of trained model classes
        elif(t[0]=='Fast_Text'):
            mod = Fast_Text.Fast_Text(dev_mode=dev_mode)  # Initialize CBOW model
            model_name = model_name_generator(t[0], t[1], training_articles)
            if model_exists_checker(model_name, dev_mode=dev_mode, language=language):
                mod.load_model(model_name, language)  # Not finished
                models.append(mod)
            else:
                print("lets train")
                i = t[1]
                mod.get_model(language=language, hs=i[0], negative=i[1], cbow_mean=i[2], iter=i[3], size=i[4], min_count=i[5],
                              max_vocab_size=i[6], workers=i[7], articles_to_learn=training_articles, randomTrain=randomTrain)  # Train model
                mod.finished_training()
                mod.save_finished_model(model_name, language)
                models.append(mod)  # Add model class to list of trained model classes
        elif (t[0] == 'Special_Fast_Text'):
            mod = Special.Fast_Text(dev_mode=dev_mode)  # Initialize CBOW model
            model_name = model_name_generator(t[0], t[1], training_articles)
            if model_exists_checker(model_name, dev_mode=dev_mode, language=language):
                mod.load_model(model_name, language)  # Not finished
                models.append(mod)
            else:
                print("lets train")
                i = t[1]
                mod.get_model(language=language, hs=i[0], negative=i[1], cbow_mean=i[2], iter=i[3], size=i[4], min_count=i[5],
                              max_vocab_size=i[6], workers=i[7], articles_to_learn=training_articles,
                              randomTrain=randomTrain)  # Train model
                mod.finished_training()
                mod.save_finished_model(model_name, language)
                models.append(mod)  # Add model class to list of trained model classes
        else:
            print(t[0]+' is not a proper word embedding algorithm which I know')
    return models
