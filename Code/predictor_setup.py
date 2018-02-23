import LeaningAlgoImpl.CBOW as CBOW
import LeaningAlgoImpl.Skip_Gram as Skip
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

def setup(leaner_list, dev_mode, training_articles=1000):
    models = []
    for t in leaner_list:
        if (t[0] == 'CBOW'):
            mod = CBOW.CBOW(dev_mode=dev_mode)  # Initialize CBOW model
            model_name = model_name_generator(t[0], t[1], training_articles)
            if model_exists_checker(model_name, dev_mode=dev_mode):
                mod.load_model(model_name)#Not finished
                models.append(mod)
            else:
                i = t[1]
                mod.get_model(hs=i[0], negative=i[1], cbow_mean=i[2], iter=i[3], size=i[4], min_count=i[5],
                          max_vocab_size=i[6], workers=i[7], articles_to_learn=training_articles)  # Train CBOW model
                mod.save_model(model_name)
                models.append(mod)  # Add model class to list of trained model classes
        if (t[0] == 'Skip_Gram'):
            mod = Skip.Skip_Gram(dev_mode=dev_mode)  # Initialize CBOW model
            model_name = model_name_generator(t[0], t[1], training_articles)
            if model_exists_checker(model_name, dev_mode=dev_mode):
                mod.load_model(model_name)#Not finished
                models.append(mod)
            else:
                i = t[1]
                mod.get_model(hs=i[0], negative=i[1], cbow_mean=i[2], iter=i[3], size=i[4], min_count=i[5],
                              max_vocab_size=i[6], workers=i[7], articles_to_learn=training_articles)  # Train model
                mod.save_model(model_name)
                models.append(mod)  # Add model class to list of trained model classes
    return models
