import LeaningAlgoImpl.CBOW as CBOW
import LeaningAlgoImpl.Skip_Gram as Skip
import os


def model_name_generator(model_type, model_params):
    name = model_type
    for param in model_params:
        name += "," + str(param)
    return name

def model_exists_checker(model_name):
    model_path = os.path.dirname(os.path.realpath(__file__)) + "/LeaningAlgoImpl/Models/"+model_name
    if(os.path.exists(model_path)):
        return True
    else:
        return False

def setup(leaner_list):
    models = []
    for t in leaner_list:
        if (t[0] == 'CBOW'):
            mod = CBOW.CBOW()  # Initialize CBOW model
            model_name = model_name_generator(t[0], t[1])
            if model_exists_checker(model_name):
                mod.load_model(model_name)#Not finished
                models.append(mod)
            else:
                i = t[1]
                mod.get_model(hs=i[0], negative=i[1], cbow_mean=i[2], iter=i[3], size=i[4], min_count=i[5],
                          max_vocab_size=i[6], workers=i[7])  # Train CBOW model
                mod.save_model(model_name_generator(t[0], i))
                print(mod)
                models.append(mod)  # Add model class to list of trained model classes
        if (t[0] == 'Skip_Gram'):
            mod = Skip.Skip_Gram()  # Initialize CBOW model
            model_name = model_name_generator(t[0], t[1])
            if model_exists_checker(model_name):
                mod.load_model(model_name)#Not finished
                models.append(mod)
            else:
                i = t[1]
                mod.get_model(hs=i[0], negative=i[1], cbow_mean=i[2], iter=i[3], size=i[4], min_count=i[5],
                              max_vocab_size=i[6], workers=i[7])  # Train model
                mod.save_model(model_name)
                print(mod)
                models.append(mod)  # Add model class to list of trained model classes
    return models
