import csv
import pandas as pd
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from matplotlib.pyplot import figure


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
    return pcc


def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max() / 2
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


def correlation_heatmap(biome_pcc, biome_names):
    """

    :param biome_pcc:
    :type biome_pcc:
    :param biome_names:
    :type biome_names:
    :return:
    :rtype:
    """
    ax = sns.heatmap(np.around(biome_pcc, decimals=2), annot=True, xticklabels=biome_names, yticklabels=biome_names)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    fig = plt.gcf()
    fig.set_size_inches(80, 60)
    plt.xticks(fontsize=20, rotation = 90)
    plt.yticks(fontsize=20)
    fig.savefig('biome_correlation_heatmap.png', dpi=100, bbox_inches='tight')
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

