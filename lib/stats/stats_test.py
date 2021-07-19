import unittest
from scipy.stats import pearsonr
from stats import *


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


if __name__ == '__main__':
    unittest.main()
