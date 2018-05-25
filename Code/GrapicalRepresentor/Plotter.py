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
    ensamble_count_array=[]
    acc_results_array = []
    for line in data:
        ensamble_count_array.append(line[0])
        temp_acc = (line[2]/(line[3]+line[2]))
        acc_results_array.append(temp_acc)
    return ensamble_count_array, acc_results_array

def file_hum_sim_unpacker(data):
    ensamble_count_array=[]
    spearman_correlation_value_array=[]
    pearson_correlation_value_array = []
    for line in data:
        ensamble_count_array.append(line[0])
        spearman_correlation_value_array.append(line[2])
        pearson_correlation_value_array.append(line[4])
    return ensamble_count_array, spearman_correlation_value_array, pearson_correlation_value_array

def r_mesurement(x_axis, y_axis):
    x_mean =0
    for x in x_axis:
        x_mean += x
    x_mean = x_mean/len(x_axis)
    y_mean = 0
    for y in y_axis:
        y_mean += y
    y_mean = y_mean / len(y_axis)
    upper = 0
    lower_1 = 0
    lower_2 = 0
    for i in range(0, len(x_axis)):
        upper += ((x_axis[i]-x_mean)*(y_axis[i]-y_mean))
        lower_1 += (x_axis[i]-x_mean)**2
        lower_2 += (y_axis[i]-y_mean)**2
    return (upper/(math.sqrt(lower_1)*math.sqrt(lower_2)))

def standard_error(r, n):
    return math.sqrt(((1-r**2)/(n-2)))

def t_value(r, se):
    return r/se

def t_test(t_mesured, n):
    t_val = t.ppf([0.95], (n-2))
    if(t_mesured > t_val):
        return True
    else:
        return False



def plot_acc_plot(data_file_name, save_file_name):
    file_parth = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/'+data_file_name
    data = np.genfromtxt(file_parth, delimiter=",", dtype=None)
    ensamble_count_array, acc_results_array = file_acc_unpacker(data)
    print(ensamble_count_array)
    print(acc_results_array)
    plt.scatter(ensamble_count_array, acc_results_array)
    plt.ylabel('Accuracy')
    plt.xlabel('#Models in ensamble')
    plt.plot(np.unique(ensamble_count_array),
             np.poly1d(np.polyfit(ensamble_count_array, acc_results_array, 1))(np.unique(ensamble_count_array)),
             color="red")
    plt.plot(np.unique(ensamble_count_array),
             np.poly1d(np.polyfit(ensamble_count_array, acc_results_array, 2))(np.unique(ensamble_count_array)),
             color="green")
    spearman_result = stats.spearmanr(ensamble_count_array, acc_results_array)
    pearson_correlation = stats.pearsonr(ensamble_count_array, acc_results_array)
    r = r_mesurement(ensamble_count_array, acc_results_array)
    se = standard_error(r, len(ensamble_count_array))
    t = t_value(r, se)
    rejected = t_test(t, len(ensamble_count_array))
    plt.text(spearman_result)
    print(spearman_result)
    print(pearson_correlation)
    print(r)
    print(se)
    print(t)
    print('Is rejected: ' + str(rejected))
    plt.savefig('Plots/'+save_file_name)
    plt.close()



def plot_hum_sim_plot(data_file_name, save_file_name):
    file_parth = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/' + data_file_name
    data = np.genfromtxt(file_parth, delimiter=",", dtype=None)
    ensamble_count_array, spearman_correlation_value_array, pearson_correlation_value_array = file_hum_sim_unpacker(data)
    plt.subplot(121)
    plt.scatter(ensamble_count_array, spearman_correlation_value_array)
    plt.ylabel('Spearman correlation')
    plt.xlabel('#Models in ensamble')
    plt.plot(np.unique(ensamble_count_array),
             np.poly1d(np.polyfit(ensamble_count_array, spearman_correlation_value_array, 1))(np.unique(ensamble_count_array)),
             color="red")
    plt.plot(np.unique(ensamble_count_array),
             np.poly1d(np.polyfit(ensamble_count_array, spearman_correlation_value_array, 2))(np.unique(ensamble_count_array)),
             color="green")
    spearman_result = stats.spearmanr(ensamble_count_array, spearman_correlation_value_array)
    pearson_correlation = stats.pearsonr(ensamble_count_array, spearman_correlation_value_array)
    r = r_mesurement(ensamble_count_array, spearman_correlation_value_array)
    se = standard_error(r, len(ensamble_count_array))
    t = t_value(r, se)
    rejected = t_test(t, len(ensamble_count_array))
    print(spearman_result)
    print(pearson_correlation)
    print(r)
    print(se)
    print(t)
    print('Is rejected: ' + str(rejected))


    plt.subplot(122)
    plt.scatter(ensamble_count_array, pearson_correlation_value_array)
    plt.ylabel('Pearson correlation')
    plt.xlabel('#Models in ensamble')
    plt.plot(np.unique(ensamble_count_array),
             np.poly1d(np.polyfit(ensamble_count_array, pearson_correlation_value_array, 1))(
                 np.unique(ensamble_count_array)),
             color="red")
    plt.plot(np.unique(ensamble_count_array),
             np.poly1d(np.polyfit(ensamble_count_array, pearson_correlation_value_array, 2))(
                 np.unique(ensamble_count_array)),
             color="green")
    spearman_result = stats.spearmanr(ensamble_count_array, pearson_correlation_value_array)
    pearson_correlation = stats.pearsonr(ensamble_count_array, pearson_correlation_value_array)
    r = r_mesurement(ensamble_count_array, pearson_correlation_value_array)
    se = standard_error(r, len(ensamble_count_array))
    t = t_value(r, se)
    rejected = t_test(t, len(ensamble_count_array))
    print(spearman_result)
    print(pearson_correlation)
    print(r)
    print(se)
    print(t)
    print('Is rejected: ' + str(rejected))
    plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
    plt.savefig('Plots/' + save_file_name)
    plt.close()



def plot_all_hardcoded():
    plot_acc_plot('Ensamble_test_results.csv', 'Ensamble_test_results')
    plot_acc_plot('tie_breaking_weighted_majority_vote.csv', 'tie_breaking_weighted_majority_vote')
    plot_acc_plot('weighted_majority_vote_Ensamble_test_results.csv', 'weighted_majority_vote_Ensamble_test_results')
    plot_hum_sim_plot('ignore_oov_human_similarity_stats.csv', 'ignore_oov_human_similarity_stats')
    plot_hum_sim_plot('naive_human_similarity_stats.csv', 'naive_human_similarity_stats')
    plot_hum_sim_plot('weight_based_on_oov_human_similarity_stats.csv', 'weight_based_on_oov_human_similarity_stats')
    plot_hum_sim_plot('weight_based_on_total_oov_ignore_oov_human_similarity_stats.csv', 'weight_based_on_total_oov_ignore_oov_human_similarity_stats')


if __name__ == "__main__": plot_hum_sim_plot('ignore_oov_human_similarity_stats.csv', 'ignore_oov_human_similarity_stats')