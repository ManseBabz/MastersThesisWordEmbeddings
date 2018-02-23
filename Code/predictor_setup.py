import LeaningAlgoImpl.CBOW as CBOW
import LeaningAlgoImpl.Skip_Gram as Skip


def model_name_generator(model_type, model_params):
    return "13"

def model_exists_checker(model_name):
    return False

def setup(leaner_list):
    models = []
    for t in leaner_list:
        if (t[0] == 'CBOW'):
            mod = CBOW.CBOW()  # Initialize CBOW model
            if model_exists_checker(model_name_generator(t[0], t[1])):
                mod.load_model("somePath")#Not finished
            else:
                i = t[1]
                mod.get_model(hs=i[0], negative=i[1], cbow_mean=i[2], iter=i[3], size=i[4], min_count=i[5],
                          max_vocab_size=i[6], workers=i[7])  # Train CBOW model
                mod.save_model(model_name_generator(t[0], i))
                models.append(mod)  # Add model class to list of trained model classes
        if (t[0] == 'Skip_Gram'):
            mod = Skip.Skip_Gram()  # Initialize CBOW model
            model_name = model_name_generator(t[0], t[1])
            if model_exists_checker(model_name):
                mod.load_model("somePath")#Not finished
            else:
                i = t[1]
                mod.get_model(hs=i[0], negative=i[1], cbow_mean=i[2], iter=i[3], size=i[4], min_count=i[5],
                              max_vocab_size=i[6], workers=i[7])  # Train model
                mod.save_model(model_name)
                models.append(mod)  # Add model class to list of trained model classes
    return models
