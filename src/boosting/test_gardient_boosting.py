import unittest
import numpy as np
import pandas as pd
from src.boosting.gradient_boosting import cart, gbrt
from src.dataset.training_dataset import TrainingDataset
from src.dataset.dataset_loader import create_data
from src.evaluators.feature_importance import get_features_importance


class TestGradientBoostingMethods(unittest.TestCase):
    # def test_cart(self):
    #     self.test_cart_1()
    #     self.test_cart_2()

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

    def nottest_print_sub_tree(self):
        x1 = np.arange(10)
        y = [1] * 5 + [10] * 5

        df = pd.DataFrame()
        df["x1"] = x1
        df["y"] = y

        tree = cart(df, 3, 3)
        tree.root.print_sub_tree()

    def nottest_gbrt_residual(self):
        x1 = np.arange(12)
        y = [1] * 3 + [3] * 3 + [10] * 3 + [20] * 3

        df = pd.DataFrame()
        df["x1"] = x1
        df["y"] = y
        dataset = TrainingDataset(df, "SalePrice")

        ensemble = gbrt(dataset, 1, 3, 4)

        self.assertEqual(ensemble.evaluate(df.iloc[0], 1), 2)
        self.assertEqual(ensemble.evaluate(df.iloc[9], 1), 15)

    def test_real_data(self):
        train, test = create_data("../../data/")

        ensemble = gbrt(train, 10, 2, 3, test_set=test)
        get_features_importance(ensemble)
        print(test)

    def test_tree_cart_evaluation(self):
        x1 = np.arange(12)
        y = [1] * 3 + [3] * 3 + [10] * 3 + [20] * 3

        df = pd.DataFrame()
        df["x1"] = x1
        df["y"] = y

        tree = cart(df, 3, 4)
        self.assertEqual(tree.evaluate(df.iloc[0]), 2)
        self.assertEqual(tree.evaluate(df.iloc[8]), 15)

    def test_tree_cart_evaluation_2(self):
        x1 = np.arange(12)
        y = [1] * 3 + [3] * 3 + [18] * 3 + [20] * 3

        df = pd.DataFrame()
        df["x1"] = x1
        df["y"] = y


        tree = cart(df, 3, 2)
        self.assertEqual(tree.evaluate(df.iloc[0]), 1)
        self.assertEqual(tree.evaluate(df.iloc[3]), 3)
        self.assertEqual(tree.evaluate(df.iloc[6]), 18)
        self.assertEqual(tree.evaluate(df.iloc[9]), 20)

if __name__ == '__main__':
    np.random.seed(125)
    unittest.main()