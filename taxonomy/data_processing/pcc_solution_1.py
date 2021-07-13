import pandas as pd
import os
import csv
import numpy as np
from scipy.stats import pearsonr
from tqdm.notebook import tqdm

## set file path
input_data_dir = '/global/cfs/cdirs/kbase/KE-Catboost/ziming/taxonomy/data'
input_file = os.path.join(input_data_dir, 'taxonomy_v4.1','feature_transpose.tsv')
output_file = os.path.join(input_data_dir, 'taxonomy_v4.1', 'feature_correlation.tsv')


def pearson_correlation_coefficient(X):
    """
    calulate pearson correlation coefficient of matrix X
    X: numpy array (MxN)
    return pcc: numpy array (MxM)
    """
    M, N = X.shape[0], X.shape[1] # number of features, number of datapoints
    X_mean = np.mean(X, axis = 1).reshape(X.shape[0], 1)
    X_std = np.std(X, axis = 1).reshape(X.shape[0], 1)
    X_tilde = (X-X_mean)/X_std
    pcc = X_tilde@X_tilde.T/N
    return pcc


def test_pearson_correlation_coefficient():
    """
    test pearson_correlation_coefficient against scipy.stats pearsonr method
    """
    ## randomly generate a 3x4 array
    test_array = np.random.rand(3,4)
    ## my pcc value matrix
    my_pcc = pearson_correlation_coefficient(test_array)
    ## initilize ground truth pcc matrix
    true_pcc = np.ones((test_array.shape[0],test_array.shape[0]))
    ## update value by scipy.stats pearsonr method
    for i in range(test_array.shape[0]):
        for j in range(test_array.shape[0]):
            if i != j:
                p, v = pearsonr(test_array[i], test_array[j])
                true_pcc[i][j] = p
                true_pcc[j][i] = p
    
    assert (np.round(my_pcc, 5) == np.round(true_pcc, 5)).all()
    print("OK!")
    
    
X = []
feature_name = []
input_file_row_count = 20059
with open(input_file, 'r') as f_in:
    reader = csv.reader(f_in, delimiter="\t")
    header = next(reader)
    print(header[:10]) 
    for i in tqdm(range(input_file_row_count)):
        row = next(reader)
        feature_name.append(row[0])
        X.append([float(_) for _ in row[1:]]) # skip feature name and convert str to float

X = np.array(X)
## calculate pcc for X
pcc_X = pearson_correlation_coefficient(X)


## write pcc matrix to a new file
with open(output_file, 'w') as f_out:
    writer = csv.writer(f_out, delimiter="\t")
    writer.writerow([None] + feature_name)
    for i in tqdm(range(len(feature_name))):
        writer.writerow([feature_name[i]]+list(pcc_X[i]))
        
print("Done!")
