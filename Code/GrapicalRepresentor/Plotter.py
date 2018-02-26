import numpy as np
import matplotlib.pyplot as plt
import textwrap as tw
import os


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
        plt.bar(X+(0.25*count), i[1], width=0.25)
        count += 1
    plot_name =generate_plot_name(List_of_pairs_of_model_and_result)
    if(check_for_plot_existance(plot_name)):
        print("This plot allready exists")
    else:
        plt.savefig('Plots/'+plot_name+'.png')


if __name__ == "__main__": create_evaluation_diagram([['CBOW', [10, 50]], ['Skip_gram', [50, 10]]])