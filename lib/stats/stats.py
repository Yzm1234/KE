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
