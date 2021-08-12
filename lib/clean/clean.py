import csv
import pandas as pd
import os
import numpy as np


def rows_and_cols_quant_filter(data, start_col_index=5, cutoff=0, pandas=True):
    """
    This method removes rows and cols whose abundance sum is below or equal the cutoff value.
    data: a pandas dataframework or numpy array
    start_col_index: the index of the first column to start with
    return: data after filtering
    """
    if pandas:
        df = data
        df['row sum'] = df.iloc[:, start_col_index:].sum(axis=1)
        if df['row sum'].max() <= cutoff:
            raise ValueError(
                "cutoff is greater than all rows' sum, data will be empty after filtering. "
                "Please choose a smaller cutoff value.")

        row_filtered_df = df.loc[df['row sum'] > cutoff]
        #         print(row_filtered_df)
        numerical_df = row_filtered_df.iloc[:, start_col_index:]
        numerical_df.loc['col sum', :] = numerical_df.sum(axis=0)
        if max(numerical_df.loc['col sum'].values[start_col_index:-1]) <= cutoff:
            raise ValueError(
                "cutoff is greater than all cols' sum, data will be empty after filtering. "
                "Please choose a smaller cutoff value.")

        numerical_df_col_filtered = numerical_df.loc[:, numerical_df.loc['col sum'].values > cutoff]
        numerical_df_col_filtered.drop(labels='col sum', axis=0, inplace=True)
        numerical_df_col_filtered.drop(labels='row sum', axis=1, inplace=True)
        res = pd.concat([row_filtered_df.iloc[:, :start_col_index], numerical_df_col_filtered], axis=1)
    else:
        pass
    return res