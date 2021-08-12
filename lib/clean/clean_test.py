import unittest
from clean import *
import pandas as pd
import numpy as np
import os


class MyTestCase(unittest.TestCase):
    data_dir = 'data'

    def test_rows_and_cols_quant_filter(self):
        test_file = os.path.join(self.data_dir, 'test.tsv')
        ground_truth_file = os.path.join(self.data_dir, 'ground_truth.tsv')
        df_test = pd.read_csv(test_file, sep="\t")
        df_ground_truth = pd.read_csv(ground_truth_file, sep="\t")
        res = rows_and_cols_quant_filter(df_test, start_col_indexgi=2, cutoff=4, pandas=True)
        self.assertTrue(df_ground_truth.reset_index(drop=True).equals(res.reset_index(drop=True)))


if __name__ == '__main__':
    unittest.main()
