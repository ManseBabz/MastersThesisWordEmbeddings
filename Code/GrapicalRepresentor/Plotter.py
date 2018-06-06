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

def file_oov_unpacker(data):
    ensamble_count_array=[]
    oov_value_array=[]
    for line in data:
        ensamble_count_array.append(line[0])
        oov_value_array.append(line[1])
    return ensamble_count_array, oov_value_array

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
        return True, t_val
    else:
        return False, t_val



def plot_acc_plot(data_file_name, save_file_name):
    file_parth = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/'+data_file_name
    data = np.genfromtxt(file_parth, delimiter=",", dtype=None)
    ensamble_count_array, acc_results_array = file_acc_unpacker(data)
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
    rejected, t_stats = t_test(t, len(ensamble_count_array))

    text_file = save_file_name+'_stats_properties'
    text_file = open('statsData/'+text_file+'.txt', "w")
    text_file.write("spearman correlation result: "+ str(spearman_result)+'\n')
    text_file.write("pearson correlation result: " + str(pearson_correlation)+'\n')
    text_file.write("r value: " + str(r)+'\n')
    text_file.write("Standard errort: " + str(se)+'\n')
    text_file.write("t-value: " + str(pearson_correlation)+'\n')
    text_file.write("t-statistics: " + str(t_stats)+'\n')
    text_file.write("Is h0 rejected?: " + str(rejected)+'\n')
    text_file.close()
    plt.savefig('Plots/'+save_file_name)
    plt.close()



def plot_hum_sim_plot(data_file_name, save_file_name):
    file_parth = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/' + data_file_name
    data = np.genfromtxt(file_parth, delimiter=",", dtype=None)
    ensamble_count_array, spearman_correlation_value_array, pearson_correlation_value_array = file_hum_sim_unpacker(data)
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
    rejected, t_stats = t_test(t, len(ensamble_count_array))
    text_file = save_file_name + '_stats_properties_spearman_results'
    text_file = open('statsData/' + text_file + '.txt', "w")
    text_file.write("spearman correlation result: " + str(spearman_result)+'\n')
    text_file.write("pearson correlation result: " + str(pearson_correlation)+'\n')
    text_file.write("r value: " + str(r)+'\n')
    text_file.write("Standard errort: " + str(se)+'\n')
    text_file.write("t-value: " + str(t)+'\n')
    text_file.write("t-statistics: " + str(t_stats)+'\n')
    text_file.write("Is h0 rejected?: " + str(rejected)+'\n')
    text_file.close()
    plt.savefig('Plots/' + save_file_name+'_spearman')
    plt.close()
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
    rejected, t_stats = t_test(t, len(ensamble_count_array))
    text_file = save_file_name + '_stats_properties_pearson_results'
    text_file = open('statsData/' + text_file + '.txt', "w")
    text_file.write("spearman correlation result: " + str(spearman_result)+'\n')
    text_file.write("pearson correlation result: " + str(pearson_correlation)+'\n')
    text_file.write("r value: " + str(r)+'\n')
    text_file.write("Standard errort: " + str(se)+'\n')
    text_file.write("t-value: " + str(t)+'\n')
    text_file.write("t-statistics: " + str(t_stats)+'\n')
    text_file.write("Is h0 rejected?: " + str(rejected)+'\n')
    text_file.close()

    plt.savefig('Plots/' + save_file_name+'_pearson')
    plt.close()

def slope_eval_plotter_acc(list_of_files, color_list, name, with_datapoints=False):
    if(len(list_of_files)==len(color_list)):
        for i in range(0, len(list_of_files)):
            file_parth = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/' + list_of_files[i]
            data = np.genfromtxt(file_parth, delimiter=",", dtype=None)
            ensamble_count_array, acc_results_array = file_acc_unpacker(data)
            if (with_datapoints):
                plt.scatter(ensamble_count_array, acc_results_array, c=color_list[i])
            plt.ylabel('Accuracy')
            plt.xlabel('#Models in ensamble')
            plt.plot(np.unique(ensamble_count_array),
                     np.poly1d(np.polyfit(ensamble_count_array, acc_results_array, 1))(np.unique(ensamble_count_array)),
                     color=color_list[i])
        plt.savefig('Plots/' + name)
        plt.close()
    else:
        raise ValueError('Inequal amount of colors and files')

def slope_eval_plotter_humsim_spearman(list_of_files, color_list, name, with_datapoints=False):
    if(len(list_of_files)==len(color_list)):
        for i in range(0, len(list_of_files)):
            file_parth = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/' + list_of_files[i]
            data = np.genfromtxt(file_parth, delimiter=",", dtype=None)
            ensamble_count_array, spearman_correlation_value_array, pearson_correlation_value_array = file_hum_sim_unpacker(
                data)
            if(with_datapoints):
                plt.scatter(ensamble_count_array, spearman_correlation_value_array, c = color_list[i])
            plt.ylabel('Spearman correlation')
            plt.xlabel('#Models in ensamble')
            plt.plot(np.unique(ensamble_count_array),
                     np.poly1d(np.polyfit(ensamble_count_array, spearman_correlation_value_array, 1))(
                         np.unique(ensamble_count_array)),
                     color=color_list[i])

        plt.savefig('Plots/' + name)
        plt.close()
    else:
        raise ValueError('Inequal amount of colors and files')

def slope_eval_plotter_humsim_pearson(list_of_files, color_list, name, with_datapoints=False):
    if (len(list_of_files) == len(color_list)):
        for i in range(0, len(list_of_files)):
            file_parth = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/' + list_of_files[i]
            data = np.genfromtxt(file_parth, delimiter=",", dtype=None)
            ensamble_count_array, spearman_correlation_value_array, pearson_correlation_value_array = file_hum_sim_unpacker(
                data)
            if (with_datapoints):
                plt.scatter(ensamble_count_array, pearson_correlation_value_array, c=color_list[i])
            plt.ylabel('Spearman correlation')
            plt.xlabel('#Models in ensamble')
            plt.plot(np.unique(ensamble_count_array),
                     np.poly1d(np.polyfit(ensamble_count_array, pearson_correlation_value_array, 1))(
                         np.unique(ensamble_count_array)),
                     color=color_list[i])

        plt.savefig('Plots/' + name)
        plt.close()
    else:
        raise ValueError('Inequal amount of colors and files')

def oov_plot(data_file_name, save_file_name):
    file_parth = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/' + data_file_name
    data = np.genfromtxt(file_parth, delimiter=",", dtype=None)
    ensamble_count_array, oov = file_oov_unpacker(
        data)
    plt.scatter(ensamble_count_array, oov)
    plt.ylabel('#OOV words')
    plt.xlabel('#Models in ensamble')

    spearman_result = stats.spearmanr(ensamble_count_array, oov)
    pearson_correlation = stats.pearsonr(ensamble_count_array, oov)
    r = r_mesurement(ensamble_count_array, oov)
    se = standard_error(r, len(ensamble_count_array))
    t = t_value(r, se)
    rejected, t_stats = t_test(t, len(ensamble_count_array))
    text_file = save_file_name + '_stats_properties_spearman_results'
    text_file = open('statsData/' + text_file + '.txt', "w")
    text_file.write("spearman correlation result: " + str(spearman_result) + '\n')
    text_file.write("pearson correlation result: " + str(pearson_correlation) + '\n')
    text_file.write("r value: " + str(r) + '\n')
    text_file.write("Standard errort: " + str(se) + '\n')
    text_file.write("t-value: " + str(t) + '\n')
    text_file.write("t-statistics: " + str(t_stats) + '\n')
    text_file.write("Is h0 rejected?: " + str(rejected) + '\n')
    text_file.close()
    plt.savefig('Plots/' + save_file_name)
    plt.close()


def plot_all_hardcoded():
    #English
    plot_acc_plot('Ensamble_test_results.csv', 'Ensamble_test_results_English')
    plot_acc_plot('tie_breaking_weighted_majority_vote.csv', 'tie_breaking_weighted_majority_vote_English')
    plot_acc_plot('weighted_majority_vote_Ensamble_test_results.csv', 'weighted_majority_vote_Ensamble_test_results_English')
    plot_acc_plot('weighted_majority_vote_Ensamble_test_results_extream.csv',
                  'weighted_majority_vote_Ensamble_test_results_English_extream')
    plot_hum_sim_plot('ignore_oov_human_similarity_stats.csv', 'ignore_oov_human_similarity_stats_English')
    plot_hum_sim_plot('naive_human_similarity_stats.csv', 'naive_human_similarity_stats_English')
    plot_hum_sim_plot('weight_based_on_oov_human_similarity_stats.csv', 'weight_based_on_oov_human_similarity_stats_English')
    plot_hum_sim_plot('weight_based_on_total_oov_ignore_oov_human_similarity_stats.csv', 'weight_based_on_total_oov_ignore_oov_human_similarity_stats_English')
    slope_eval_plotter_acc(['Ensamble_test_results.csv',
                           'tie_breaking_weighted_majority_vote.csv',
                           'weighted_majority_vote_Ensamble_test_results.csv'],
                          ['red', 'blue', 'green'],
                          'allSloaps_acc_English')
    slope_eval_plotter_acc(['Ensamble_test_results.csv',
                           'tie_breaking_weighted_majority_vote.csv',
                           'weighted_majority_vote_Ensamble_test_results.csv'],
                           ['red', 'blue', 'green'],
                           'allSloaps_acc_English_data', with_datapoints=True)
    slope_eval_plotter_acc(['Ensamble_test_results.csv',
                            'tie_breaking_weighted_majority_vote.csv',
                            'weighted_majority_vote_Ensamble_test_results_extream.csv'],
                           ['red', 'blue', 'green'],
                           'allSloaps_acc_English_extream', with_datapoints=False)
    slope_eval_plotter_acc(['Ensamble_test_results.csv',
                            'tie_breaking_weighted_majority_vote.csv',
                            'weighted_majority_vote_Ensamble_test_results_extream.csv'],
                           ['red', 'blue', 'green'],
                           'allSloaps_acc_English_data_extream', with_datapoints=True)
    slope_eval_plotter_humsim_spearman(['ignore_oov_human_similarity_stats.csv',
                                       'naive_human_similarity_stats.csv',
                                       'weight_based_on_oov_human_similarity_stats.csv',
                                       'weight_based_on_total_oov_ignore_oov_human_similarity_stats.csv'],
                                      ['red', 'blue', 'black', 'green'],
                                      'allSloaps_spearman_English')
    slope_eval_plotter_humsim_spearman(['ignore_oov_human_similarity_stats.csv',
                                       'naive_human_similarity_stats.csv',
                                       'weight_based_on_oov_human_similarity_stats.csv',
                                       'weight_based_on_total_oov_ignore_oov_human_similarity_stats.csv'],
                                      ['red', 'blue', 'black', 'green'],
                                      'allSloaps_spearman_English_data', with_datapoints=True)
    slope_eval_plotter_humsim_pearson(['ignore_oov_human_similarity_stats.csv',
                                       'naive_human_similarity_stats.csv',
                                       'weight_based_on_oov_human_similarity_stats.csv',
                                       'weight_based_on_total_oov_ignore_oov_human_similarity_stats.csv'],
                                      ['red', 'blue', 'black', 'green'],
                                      'allSloaps_pearson_English')
    slope_eval_plotter_humsim_pearson(['ignore_oov_human_similarity_stats.csv',
                                       'naive_human_similarity_stats.csv',
                                       'weight_based_on_oov_human_similarity_stats.csv',
                                       'weight_based_on_total_oov_ignore_oov_human_similarity_stats.csv'],
                                      ['red', 'blue', 'black', 'green'],
                                      'allSloaps_pearson_English_data', with_datapoints=True)
    oov_plot('oov_test.csv', 'oov_test_english')

    #Danish
    """
    plot_acc_plot('Ensamble_test_results_danish.csv', 'Ensamble_test_results_Danish')
    plot_acc_plot('tie_breaking_weighted_majority_vote_danish.csv', 'tie_breaking_weighted_majority_vote_Danish')
    plot_acc_plot('weighted_majority_vote_Ensamble_test_results_danish.csv',
                  'weighted_majority_vote_Ensamble_test_results_Danish')
    plot_hum_sim_plot('ignore_oov_human_similarity_stats_danish.csv', 'ignore_oov_human_similarity_stats_Danish')
    plot_hum_sim_plot('naive_human_similarity_stats_danish.csv', 'naive_human_similarity_stats_Danish')
    plot_hum_sim_plot('weight_based_on_oov_human_similarity_stats_danish.csv',
                      'weight_based_on_oov_human_similarity_stats_Danish')
    plot_hum_sim_plot('weight_based_on_total_oov_ignore_oov_human_similarity_stats_danish.csv',
                      'weight_based_on_total_oov_ignore_oov_human_similarity_stats_Danish')
    slope_eval_plotter_acc(['Ensamble_test_results.csv',
                            'tie_breaking_weighted_majority_vote.csv',
                            'weighted_majority_vote_Ensamble_test_results.csv'],
                           ['red', 'blue', 'green'],
                           'allSloaps_acc_Danish')
    slope_eval_plotter_acc(['Ensamble_test_results.csv',
                            'tie_breaking_weighted_majority_vote.csv',
                            'weighted_majority_vote_Ensamble_test_results.csv'],
                           ['red', 'blue', 'green'],
                           'allSloaps_acc_Danish_data', with_datapoints=True)
    slope_eval_plotter_humsim_spearman(['ignore_oov_human_similarity_stats_danish.csv',
                                       'naive_human_similarity_stats_danish.csv',
                                       'weight_based_on_oov_human_similarity_stats_danish.csv',
                                       'weight_based_on_total_oov_ignore_oov_human_similarity_stats_danish.csv'],
                                       ['red', 'blue', 'black', 'green'],
                                       'allSloaps_pearson_Danish')
    slope_eval_plotter_humsim_spearman(['ignore_oov_human_similarity_stats_danish.csv',
                                       'naive_human_similarity_stats_danish.csv',
                                       'weight_based_on_oov_human_similarity_stats_danish.csv',
                                       'weight_based_on_total_oov_ignore_oov_human_similarity_stats_danish.csv'],
                                       ['red', 'blue', 'black', 'green'],
                                       'allSloaps_pearson_Danish_data', with_datapoints=True)
    slope_eval_plotter_humsim_pearson(['ignore_oov_human_similarity_stats_danish.csv',
                                       'naive_human_similarity_stats_danish.csv',
                                       'weight_based_on_oov_human_similarity_stats_danish.csv',
                                       'weight_based_on_total_oov_ignore_oov_human_similarity_stats_danish.csv'],
                                      ['red', 'blue', 'black', 'green'],
                                      'allSloaps_pearson_Danish')
    slope_eval_plotter_humsim_pearson(['ignore_oov_human_similarity_stats_danish.csv',
                                       'naive_human_similarity_stats_danish.csv',
                                       'weight_based_on_oov_human_similarity_stats_danish.csv',
                                       'weight_based_on_total_oov_ignore_oov_human_similarity_stats_danish.csv'],
                                      ['red', 'blue', 'black', 'green'],
                                      'allSloaps_pearson_Danish_data', with_datapoints=True)
    """



if __name__ == "__main__": plot_all_hardcoded()