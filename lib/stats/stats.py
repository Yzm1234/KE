import csv
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from pathlib import Path
from matplotlib.pyplot import figure
from ..plot import plot


def pearson_correlation_coefficient(X):
    """
    calculate pearson correlation coefficient of matrix X
    X: numpy array (MxN)
    return pcc: numpy array (MxM)
    """
    M, N = X.shape[0], X.shape[1]  # number of features, number of data points
    X_mean = np.mean(X, axis=1).reshape(X.shape[0], 1)
    X_std = np.std(X, axis=1).reshape(X.shape[0], 1)
    X_tilde = (X-X_mean)/X_std
    pcc = X_tilde@X_tilde.T/N
    np.fill_diagonal(pcc, 1, wrap=False)
    return pcc


def cluster_corr(corr_array, distance_method=None, absolute_value=True, threshold=None, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix
    threshold:
        :param threshold: the threshold of distance when doing clustering,
                            the maximum inter-cluster distance allowed
        :type threshold: float between 0-1
    absolute_value:
        :param : if using absolute value of correlation
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    if not distance_method:
        pairwise_distances = sch.distance.pdist(corr_array)#, metric='correlation')
    elif distance_method == "oneminus":
        if absolute_value:
            pairwise_distances = squareform(1 - abs(corr_array))
        else:
            pairwise_distances = squareform(1 - corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    if not threshold:
        cluster_distance_threshold = pairwise_distances.max() / 2
    else:
        cluster_distance_threshold = threshold
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


def feature_extraction(pcc_mat, cutoff):
    """
    AKA: feature_selection_method_4 in notebooks
    select features from full pearson correlation coefficient (PCC) matrix whose correlation is below cutoff
    pcc_mat:
    cutoff: pcc value threshold
    return: a list of feature mask, True means selected, False means not selected
    """
    M = pcc_mat.shape[0]  # number of features
    feature_mask = [True] * M
    pcc_mat_abs = np.abs(pcc_mat)
    for i in range(M):
        for j in range(i):
            if pcc_mat_abs[i][j] >= cutoff:
                pcc_mat_abs[i, :] = 0
                pcc_mat_abs[:, i] = 0
                feature_mask[i] = False
    return feature_mask


def generate_symmetric_matrix(low_bound, high_bound, shape):
    """
    This method randomly generates a symmetric matrix
    low_bound: the minimum number in matrix
    high_bound: the maximum number in matrix
    shape: a tuple (wide, height) of matrix
    """
    b = np.random.uniform(low_bound, high_bound, (shape[0], shape[1]))
    b_symm = (b + b.T)/2
    np.fill_diagonal(b_symm, 0)
    b_symm
    return b_symm


def barh_counter(input_list):
    """
    This method takes a list and plot bar plot of its items' frequency distribution, no tests
    :param input_list: the target list of items need to be ploted
    :type input_list: list
    :return: a horizontal bar plot
    :rtype: figure
    """
    counter = Counter(input_list)
    labels, values = zip(*sorted(counter.items(), reverse=False, key=lambda x:x[1]))
    figure(figsize=(10, 8), dpi=500)
    plt.barh(range(len(labels)), values)
    plt.yticks(range(len(labels)), labels, rotation=0, fontsize=6)
    plt.show()


def clustered_correlation_heatmap(corr_matrix, save_figure=True, figure_name="clustered_correlation_heatmap.png"):
    """
    This method takes in a clustered correlation matrix and return a heatmap figure
    :param corr_matrix: clustered correlatino matrix
    :type corr_matrix: pandas data frame
    :param save_figure: if save the heatmap
    :type save_figure: boolean
    :param figure_name: name of saved figure
    :type figure_name: string
    :return: an image
    :rtype: image
    """
    ax = sns.heatmap(np.around(corr_matrix, decimals=2), annot=True,
                     cmap='coolwarm', vmin=-1, vmax=1,
                     xticklabels=corr_matrix.index, yticklabels=corr_matrix.index)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    fig = plt.gcf()
    fig.set_size_inches(80, 60)
    plt.xticks(fontsize=20, rotation = 90)
    plt.yticks(fontsize=20)
    if save_figure:
        fig.savefig(figure_name, dpi=100, bbox_inches='tight')
    plt.show()


def pairwise_correlation(names_list, pcc_mat):
    """
    This method generates pair wise correlation table from pearson correlation coefficient matrix
    :param names_list: the name list of the objects
    :type names_list:  list
    :param pcc_mat:  pearson correlation coefficient matrix
    :type pcc_mat: numpy array
    :return: a list of pair wise correlation value: ['object_1','object_2', 'pearson correlation'], first row is header
    :rtype: list
    """
    res = [['object_1', 'object_2', 'pearson correlation']]
    for i in range(len(names_list)-1):
        for j in range(i+1, len(names_list)):
            v = pcc_mat[i][j]
            obj_1 = names_list[i]
            obj_2 = names_list[j]
            res.append([obj_1, obj_2, v])
            res.append([obj_2, obj_1, v])
    return res


def top_prob_labels(class_list, sample, prob_array, num_prob=None, plot=False, num_bar=10):
    """
    This method gives num most likely labels based on probability and plot the bar char
    :param class_list: the model class name in order
    :type class_list: list
    :param sample: test sample name list
    :type sample: list
    :param prob_array: probabilities of every sample being classified as every class
    :type prob_array: 2d numpy array of shape (N*C) N is the number of data points, C is the number of classes
    :param num_prob: the number of most likely classes
    :type num_prob: int
    :param plot: if plot bar char for top labels
    :type plot: boolean
    :param num_bar: number of bar in each plot
    :type num_bar: int
    :return: result in format [(class1, p1), (class2, p2), (class3, p3)] if num == 3
    :rtype: list
    """
    if not num_prob:
        num_prob = len(class_list)
    if num_prob > len(class_list):
        raise ValueError('Number of top possible labels should be no more than total number of classes')
    if len(sample) != prob_array.shape[0]:
        raise ValueError("Number of samples and probability matrix row don't match")
    res = []
    for row in prob_array:
        sorted_idx = np.argsort(row)[::-1]  # from highest to lowest
        res.append([(class_list[idx], row[idx]) for idx in sorted_idx[:num_prob]])
    if plot:
        Path("top_labels_bar_chart/").mkdir(parents=True, exist_ok=True)
        for i, pair in enumerate(res):
            label, prob = zip(*pair[:num_bar])
            plt.barh(label, prob)
            plt.title('{}'.format(sample[i]))
            plt.gca().invert_yaxis()
            plt.xlabel('probability')
            plt.savefig("top_labels_bar_chart/{}.png".format(sample[i]), bbox_inches='tight')
            plt.show()
    return res


def get_macro_and_weight_score(auc_dict, weight_dict):
    """
    This methods takes an auc_dict and weight_dic and return the macro and weighted average auc score
    :param auc_dict: a dictionary (key: label, value: auc score of the label)
    :type auc_dict: dictionary
    :param weight_dict: a dictionary (key: label, value: weight of the label)
    :type weight_dict: dictionary
    :return: macro auc score and weighted auc score
    :rtype: tuple
    """
    macro_auc = np.mean([score for _, score in auc_dict.items()])
    weighted_auc = 0
    for label, auc_score in auc_dict.items():
        weighted_auc += auc_dict[label] * weight_dict[label]
    return macro_auc, weighted_auc


def get_weight(y_test):
    weight = {}
    counter = Counter(y_test)
    for label, cnt in counter.items():
        weight[label] = cnt/len(y_test)
    return weight


def insert_roc_pr(report, roc_auc, pr_auc, weight):
    """
    This method insert roc auc score and pr auc score columns to original sklearn.metrics generating classification_report
    :param report: a report dataframe work
    :type report: pandas dataframe work
    :param roc_auc: each label's roc auc score
    :type roc_auc: dictionary
    :param pr_auc: each label's pr auc score
    :type pr_auc: dictionary
    :param weight: each label's weight
    :type weight: dictionary
    :return: report
    :rtype: pandas dataframe work
    """
    macro_roc, weighted_roc = get_macro_and_weight_score(roc_auc, weight)
    macro_pr, weighted_pr = get_macro_and_weight_score(pr_auc, weight)
    roc_list = [roc_auc[label] for label in report.index[:-3]] + [report.loc['accuracy'][0]] + [macro_roc, weighted_roc]
    pr_list = [pr_auc[label] for label in report.index[:-3]] + [report.loc['accuracy'][0]] + [macro_pr, weighted_pr]
    report.insert(3, 'roc-auc', roc_list)
    report.insert(4, 'pr-auc', pr_list)
    return report


def save_report(report):
    report.support = report.support.astype(int)
    report = report.round(3)
    report.to_csv("model_analysis_result/score_report.tsv", sep="\t")


def model_analysis(model, X_test, y_test):
    """
    This method will output all analysis results together (ROC, PR, confusino matrix, feature importance)
    :param model: Catboost model
    :type model: Catboost model object
    :param X_test: Test set
    :type X_test: Pandas dataframe work
    :param y_test: True lables of test set
    :type y_test: list or pandas series
    :return: score report
    :rtype: pd framework
    """
    y_pred_prob = model.predict(X_test, prediction_type="Probability")
    y_pred_class = model.predict(X_test)
    print("ROC curve:\n", "=" * 100)
    roc_auc = plot.roc(y_test, y_pred_prob, model.classes_, show_plots=False)
    print("PR curve:\n", "=" * 100)
    pr_auc = plot.pr(y_test, y_pred_prob, model.classes_, show_plots=False)
    report = pd.DataFrame(classification_report(y_test, y_pred_class, output_dict=True)).transpose()
    weight = get_weight(y_test)
    report = insert_roc_pr(report, roc_auc, pr_auc, weight)
    save_report(report)
    print("Confusion matrix:\n", "=" * 100)
    plot.confusion_matrix(model, X_test, y_test)
    print("Top probable labels:\n", "=" * 100)
    top_prob_labels(model.classes_, list(y_test), y_pred_prob, num_prob=5, plot=True, num_bar=8)
    return report


def feature_selection(df, cutoff=0.9):
    """
    This method takes features table and filters out features util correlation of any pair is below the cutoff
    :param df: feature table, N x (M + 1) M: number of features plus one label column ('biome')
    :type df: pandas dataframe
    :param cutoff:  when a pair of features correlation coefficient number is above this threshold, the feature
                    with smaller index will be removed
    :type cutoff: float
    :return: feature table after removing highly correlated features
    :rtype: pandas dataframe work
    """
    df = df.set_index('biome')
    feature_mat = df.to_numpy().transpose()
    pcc_mat = pearson_correlation_coefficient(feature_mat)
    feature_mask = feature_extraction(pcc_mat, 0.9)
    selected_features = df.columns[feature_mask]
    df = df[selected_features]
    df.insert(0, 'biome', df.index, True)
    df = df.reset_index(drop=True)
    return df


def get_high_present_samples(df, first_numerical_col_idx, threshold):
    df.reset_index(drop=True, inplace=True)
    bin_df = df.iloc[:, first_numerical_col_idx:]
    bin_feature = np.where(bin_df.to_numpy()>0, 1, 0)
    bin_df[:] = bin_feature
    present_sum = bin_df.sum(axis=1)
    great = []
    for i, n in enumerate(present_sum):
        if n >= threshold:
            great.append(i)
    df = df.filter(items=great, axis=0)
    return df

