import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from numpy.polynomial.polynomial import polyfit
import textwrap as tw
from scipy.stats import t
from scipy import stats
import math
import os

"""
    Tabeller (se efficient estimation artikel)
    Acroracy and micro-F1 i forhold til train size
    precision/recall curve (Dependency-based word embeddings)
    Score barcharts (Evaluation methods artikkel)
    Avg word rank by frequency (Evaluation methods)
    How well does the embedding determine if a word is frequent or rare (Evaluation methods) How?
"""
def file_acc_unpacker(data):
    model_name =[]
    acc_results_array = []
    for line in data:
        model_name.append(line[0])
        temp_acc = (line[8]/(line[9]+line[8]))
        acc_results_array.append(temp_acc)
    return model_name, acc_results_array

def file_hum_sim_unpacker(data):
    model_name = []
    spearman_correlation_value_array=[]
    pearson_correlation_value_array = []
    for line in data:
        model_name.append(line[0])
        spearman_correlation_value_array.append(line[9])
        pearson_correlation_value_array.append(line[11])
    return model_name, spearman_correlation_value_array, pearson_correlation_value_array

def file_clustering_unpacker(data):
    model_name = []
    acc_results_array = []
    for line in data:
        model_name.append(line[0])
        temp_acc = (line[8] / (line[9] + line[8]))
        acc_results_array.append(temp_acc)
    return model_name, acc_results_array




def plot_acc_plot(data_file_name, save_file_name):
    file_parth = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/'+data_file_name
    data = np.genfromtxt(file_parth, delimiter=",", dtype=None)
    ensamble_count_array, acc_results_array = file_acc_unpacker(data)
    plt.scatter(ensamble_count_array, acc_results_array)
    plt.ylabel('Accuracy')
    plt.xlabel('Model type')


    plt.savefig('Plots/'+save_file_name)
    plt.close()


def plot_hum_sim_plot(data_file_name, save_file_name):
    file_parth = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/' + data_file_name
    data = np.genfromtxt(file_parth, delimiter=",", dtype=None)
    ensamble_count_array, spearman_correlation_value_array, pearson_correlation_value_array = file_hum_sim_unpacker(data)
    plt.scatter(ensamble_count_array, spearman_correlation_value_array)
    plt.ylabel('Spearman correlation')
    plt.xlabel('Model type')

    plt.savefig('Plots/' + save_file_name+'_spearman')
    plt.close()

    plt.scatter(ensamble_count_array, pearson_correlation_value_array)
    plt.ylabel('Pearson correlation')
    plt.xlabel('Model type')


    plt.savefig('Plots/' + save_file_name+'_pearson')
    plt.close()

def plot_clustering_plot(data_file_name, save_file_name):
    file_parth = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/'+data_file_name
    data = np.genfromtxt(file_parth, delimiter=",", dtype=None)
    ensamble_count_array, acc_results_array = file_acc_unpacker(data)
    plt.scatter(ensamble_count_array, acc_results_array)
    plt.ylabel('Accuracy')
    plt.xlabel('Model type')


    plt.savefig('Plots/'+save_file_name)
    plt.close()



def plot_all_hardcoded():
    plot_acc_plot('individual_models_acc_Danish.csv', 'individual_models_acc_danish')

    plot_hum_sim_plot('individual_models_human_similarity_Danish.csv','individual_models_human_similarity_danish')

    plot_clustering_plot('individual_models_clustering_DanishFrugt-dyr-køretøjer.csv.csv',
                         'individual_models_clustering_danishFrugt-dyr-køretøjer')
    plot_clustering_plot('individual_models_clustering_DanishHus-værktøj-kropsdele.csv.csv',
                         'individual_models_clustering_danishHus-værktøj-kropsdele')
    plot_clustering_plot('individual_models_clustering_DanishNavneord-udsagnsord-tillægsord.csv.csv',
                         'individual_models_clustering_danishNavneord-udsagnsord-tillægsord')



if __name__ == "__main__": plot_all_hardcoded()




























