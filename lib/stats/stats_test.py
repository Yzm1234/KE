import unittest
from scipy.stats import pearsonr
from stats import *
import numpy as np


class TestStats(unittest.TestCase):
    def test_pearson_correlation_coefficient(self):
        """
        test pearson_correlation_coefficient against scipy.stats pearsonr method
        """
        # randomly generate a 3x4 array
        test_array = np.random.rand(3, 4)
        # my pcc value matrix
        my_pcc = pearson_correlation_coefficient(test_array)
        # initialize ground truth pcc matrix
        true_pcc = np.ones((test_array.shape[0], test_array.shape[0]))
        # update value by scipy.stats pearsonr method
        for i in range(test_array.shape[0]):
            for j in range(test_array.shape[0]):
                if i != j:
                    p, v = pearsonr(test_array[i], test_array[j])
                    true_pcc[i][j] = p
                    true_pcc[j][i] = p
        self.assertEqual(np.round(my_pcc, 5).all(), np.round(true_pcc, 5).all())

    def test_feature_selection_method(self):
        """
        This test checks if each pair features' pearson correlation is below the threshold
        """
        pcc = generate_symmetric_matrix(0.75, 1, (10, 10))
        feature_mask = feature_extraction(pcc, 0.9)
        true_idx = np.where(feature_mask)[0]  # return index of True item in feature_mask
        for i in range(len(true_idx) - 1):
            for j in range(i + 1, len(true_idx)):
                r, c = true_idx[i], true_idx[j]
                self.assertTrue(pcc[r, c] < 0.9)

    def test_pairwise_correlation(self):
        test_pcc = np.array([[0., 0.84, 0.9],
                            [0.84, 0., 0.87],
                            [0.9, 0.87, 0.]])

        names = ['A', 'B', 'C']
        res = pairwise_correlation(names, test_pcc)
        target = [['object_1', 'object_2', 'pearson correlation'],
                  ['A', 'B', 0.84],
                  ['B', 'A', 0.84],
                  ['A', 'C', 0.9],
                  ['C', 'A', 0.9],
                  ['B', 'C', 0.87],
                  ['C', 'B', 0.87]]
        self.assertEqual(res, target)


if __name__ == '__main__':
    unittest.main()
