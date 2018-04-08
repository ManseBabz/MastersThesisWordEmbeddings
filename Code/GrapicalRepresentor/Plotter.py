import numpy as np
import matplotlib.pyplot as plt
import textwrap as tw
import os

"""
    Tabeller (se efficient estimation artikel)
    Acroracy and micro-F1 i forhold til train size
    precision/recall curve (Dependency-based word embeddings)
    Score barcharts (Evaluation methods artikkel)
    Avg word rank by frequency (Evaluation methods)
    How well does the embedding determine if a word is frequent or rare (Evaluation methods) How?
"""
def check_for_plot_existance(plt_name):
    plt_parth =  os.path.dirname(os.path.realpath(__file__)) + '/Plots/'+plt_name+'.png'
    if(os.path.exists(plt_parth)):
        return True
    else:
        return False

def generate_plot_name(List_of_model_names):
    result='plot'
    for name in List_of_model_names:
        result += '_'+name[0]
    return result

def create_evaluation_diagram(List_of_pairs_of_model_and_result):
    count = 0
    for i in List_of_pairs_of_model_and_result:
        X = np.arange(len(i[1]))
        #Generate evaluation data
        #Create prober plot
        count += 1
    plot_name =generate_plot_name(List_of_pairs_of_model_and_result)
    if(check_for_plot_existance(plot_name)):
        print("This plot allready exists")
    else:
        plt.savefig('Plots/'+plot_name+'.png')

def micro_f1_plot():
    print("not implemented yet")

def accuracy_for_training_size_plot():
    print("not implemented yet")

def precision_recall_corelation():
    print("not implemented yet")

def score_bar_chart():
    print("not implemented yet")

def avg_word_rank_by_frequency():
    print("not implemented yet")


if __name__ == "__main__": create_evaluation_diagram([['CBOW', [10, 50]], ['Skip_gram', [50, 10]]])