import csv
import pandas as pd
import os
import numpy as np


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