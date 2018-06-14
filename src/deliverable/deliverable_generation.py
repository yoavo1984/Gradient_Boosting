# Standard libraries.
import pickle

# Local libraries.
from src.boosting.boosting_hyperparameters import GBRTHyperparameters
from src.boosting.gradient_boosting import gbrt

DEPTH_OPTIONS = [2,3,4]


def iterate_depth_parameter(train_set, test_set):
    hyperparameters = GBRTHyperparameters(2, 5, 3, 1, 0.5, 0)
    results = {}

    for depth in DEPTH_OPTIONS:
        errors = []
        hyperparameters.max_depth = depth
        gbrt(train_set, hyperparameters, test_set, errors)
        results[depth] = errors

    with open('depth_deliverable.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_deliverable(train_set, test_set):
    iterate_depth_parameter(train_set, test_set)