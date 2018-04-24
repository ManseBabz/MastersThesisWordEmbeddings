import EnsamblePredictor as EP
import logging, os

def question_word_test():
    ensamble = EP.simple_ensamble([['CBOW',[0,5,0,10,100,1,3000000]], ['CBOW',[0,5,0,10,100,1,3000000]] ])
    dir_path = os.path.dirname(os.path.realpath(__file__))+"/TestingSet/wordsim353.tsv"
    ensamble.set_weights([1,0.5])
    ensamble.evaluate_word_pairs(dir_path, similarity_model_type=0)
    #simple_ensamble(["CBOW,0,5,0,10,100,1,3000000", "CBOW,0,5,0,10,100,1,5000000", "CBOW,0,5,0,10,100,1,7000000"])

def word_sim_test():
    print("not implemented")

if __name__ == "__main__": question_word_test()