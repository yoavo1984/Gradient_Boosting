import unittest
import numpy as np
import pandas as pd
from src.boosting.gradient_boosting import cart


class TestGradientBoostingMethods(unittest.TestCase):
    def test_cart(self):
        self.test_cart_1()
        self.test_cart_2()

    def test_cart_1(self):
        x1 = np.arange(10)
        y = [1] * 5 + [10] * 5

        df = pd.DataFrame()
        df["x1"] = x1
        df["y"] = y

        tree = cart(df, 3, 3)
        self.assertEqual(tree.root.s, 4)
        self.assertEqual(tree.root.left_descendant.const, 1)
        self.assertEqual(tree.root.right_descendant.const, 10)

    def test_cart_2(self):
        x1 = np.arange(12)
        y = [1] * 3 + [3] * 3 + [10] * 3 + [20] * 3

        df = pd.DataFrame()
        df["x1"] = x1
        df["y"] = y

        tree = cart(df, 3, 4)
        self.assertEqual(tree.root.s, 5)
        self.assertEqual(tree.root.left_descendant.const, 2)
        self.assertEqual(tree.root.right_descendant.const, 15)

if __name__ == '__main__':
    unittest.main()