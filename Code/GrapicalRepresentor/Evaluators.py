import os



def evaluate_using_questions_words(model):
    correct = 0
    wrong =0
    with open(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/TestingSet/questions-words.txt') as f:
        for line in f:
            words = line.split(" ")
            if(model.evaluate() == words[3]):
                correct += 1
            else:
                wrong +=1
    return correct, wrong




def plural_evaluate_using_questions_words(list_of_models):
    correct = []
    wrong = []
    with open(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/TestingSet/questions-words.txt') as f:
        for line in f:
            words = line.split(" ")
            for eval_index in range(0, len(list_of_models)):
                if (list_of_models[eval_index].evaluate() == words[3]):
                    correct[eval_index] += 1
                else:
                    wrong[eval_index] += 1
    return correct, wrong

def f1_mesure():
    pre = precision()
    rec = recall()
    f1 = 2*((pre*rec)/(pre+rec))
    return f1

def precision():
    print("Not implemented yet")
    return 0

def recall():
    print("Not implemented yet")
    return 0

def Avg_word_rank():
    print("Not implemented yet")

def rare_or_non_rare_word():
    print("Not implemented yet")

