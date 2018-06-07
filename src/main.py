from src.dataset.dataset_loader import create_data

path_to_file = "data"

if __name__ == "__main__":
    train, test = create_data(path_to_file)
    print (train.dataframe.info())
    print (test.dataframe.info())