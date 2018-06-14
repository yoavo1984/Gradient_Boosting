import time

from src.boosting.boosting_hyperparameters import GBRTHyperparameters
from src.boosting.gradient_boosting import gbrt
from src.dataset.dataset_loader import create_data

path_to_file = "../data/"


def parse_configuration_file():
    with open("configuration") as configuration_file:
        lines = configuration_file.readlines()
        configuration = lines[1].split(", ")

    return GBRTHyperparameters(*configuration,)


if __name__ == "__main__":
    # Get train,test dataset and hyperparameters from file.
    train, test = create_data(path_to_file)
    hyperparameters = parse_configuration_file()

    # Clearing out previous file.
    with open("error.txt", "w"):
        pass

    # Perform training.
    start = time.time()
    ensemble = gbrt(train, hyperparameters, test)
    training_time = time.time() - start

    # Get training statistics.
    train_rmse = ensemble.compute_dataset_rmse(train.get_dataframe_copy())
    test_rmse = ensemble.compute_dataset_rmse(test.get_dataframe_copy())

    # Write hyperparameters, statistics and training time to file.
    with open("outcome.txt", "w") as outcome_file:
        outcome_file.write("HyperParameters:\n" + "="*20 + "\n")
        outcome_file.write("{}".format(str(hyperparameters)))

        outcome_file.write("\n\nErrors:\n" + "="*20)
        outcome_file.write("\nTrain error = {}".format(train_rmse))
        outcome_file.write("\nTest error = {}".format(test_rmse))

        outcome_file.write("\n\nRunningTime:\n" + "="*20 + "\n")
        outcome_file.write("{}".format(training_time))


