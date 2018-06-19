# Standard libraries.
import pickle

# Local libraries.
import time

from src.boosting.boosting_hyperparameters import GBRTHyperparameters
from src.boosting.gradient_boosting import gbrt

DEPTH_OPTIONS = [2,3,4]
THRESHOLD_OPTIONS = [1, 3, 5, 10, 20]
SAMPLING_OPTIONS = [0.1, 0.3, 0.5, 0.8, 1]

DEFAULT_HYPERPARAMETERS = [100, 5, 3, 0.1, 0.3, 0]


def iterate_depth_parameter(train_set, test_set):
    hyperparameters = GBRTHyperparameters(*DEFAULT_HYPERPARAMETERS,)
    results = {}

    for depth in DEPTH_OPTIONS:
        errors = []
        hyperparameters.max_depth = depth
        tree = gbrt(train_set, hyperparameters, test_set, errors)
        results[depth] = errors[:]

    with open('depth_deliverable.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def iterate_threshold_parameter(train_set, test_set):
    hyperparameters = GBRTHyperparameters(*DEFAULT_HYPERPARAMETERS,)
    results = {}
    results["time"] = []

    for threshold in THRESHOLD_OPTIONS:
        errors = []
        hyperparameters.num_threshold = threshold

        start = time.time()
        gbrt(train_set, hyperparameters, test_set, errors)
        training_time = time.time() - start

        results[threshold] = errors[:]
        results["time"].append((threshold, training_time))

    with open('threshold_deliverable.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def iterate_sample_parameter(train_set, test_set):
    hyperparameters = GBRTHyperparameters(*DEFAULT_HYPERPARAMETERS,)
    results = {}
    results["time"] = []

    for  sampling_portion in SAMPLING_OPTIONS:
        errors = []
        hyperparameters.sampling_portion = sampling_portion

        start = time.time()
        gbrt(train_set, hyperparameters, test_set, errors)
        training_time = time.time() - start

        results[sampling_portion] = errors[:]
        results["time"].append((sampling_portion, training_time))

    with open('sampling_deliverable.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_deliverable(train_set, test_set):
    iterate_depth_parameter(train_set, test_set)
    iterate_threshold_parameter(train_set, test_set)
    iterate_sample_parameter(train_set, test_set)
