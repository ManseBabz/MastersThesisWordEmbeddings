import os
class singular_evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate_using_questions_words(self):
        correct = 0
        wrong =0
        with open(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/TestingSet/questions-words.txt') as f:
            for line in f:
                words = line.split(" ")
                if(self.model.evaluate() == words[3]):
                    correct += 1
                else:
                    wrong +=1
        return correct, wrong


class plural_evaluater():
    def __init__(self, list_of_models):
        self.model = list_of_models

    def evaluate_using_questions_words(self):
        correct = []
        wrong = []
        with open(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/TestingSet/questions-words.txt') as f:
            for line in f:
                words = line.split(" ")
                for eval_index in range(0, len(self.list_of_models)):
                    if (self.model.evaluate() == words[3]):
                        correct[eval_index] += 1
                    else:
                        wrong[eval_index] += 1
        return correct, wrong




if __name__ == "__main__": singular_evaluator.evaluate_using_questions_words()