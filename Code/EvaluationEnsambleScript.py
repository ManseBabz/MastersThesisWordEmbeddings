import EnsamblePredictor as EP
import logging, os

def question_word_test():
    ensamble = EP.simple_ensamble([['CBOW',[0,5,0,10,100,1,3000000]], ['CBOW',[0,5,0,10,100,1,3000000]] ])
    dir_path = os.path.dirname(os.path.realpath(__file__))+"/TestingSet/questions-words.txt"
    ensamble.accuracy(dir_path, predictor_method=1)
        #simple_ensamble(["CBOW,0,5,0,10,100,1,3000000", "CBOW,0,5,0,10,100,1,5000000", "CBOW,0,5,0,10,100,1,7000000"])

def word_sim_test():
    print("not implemented")

if __name__ == "__main__": question_word_test()