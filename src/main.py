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
    train, test = create_data(path_to_file)
    hyperparameters = parse_configuration_file()
    gbrt(train, hyperparameters, test)
