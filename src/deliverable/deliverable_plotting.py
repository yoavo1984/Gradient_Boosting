import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_depth_deliverable():
    # Load pickle data object
    with open('depth_deliverable.pickle', 'rb') as handle:
        data = pickle.load(handle)

    # Plot depth vs error
    for value, errors in data.items():
        num_of_trees = np.arange(len(errors))

        plt.title("Tree Depth = {}".format(value))

        plt.xlabel("Number of trees")
        plt.ylabel("RMSE")
        plt.plot(num_of_trees, errors)
        plt.legend(["Train", "Test"])
        plt.savefig("{}_depth.png".format(value))
        plt.clf()


def plot_sampling_deliverable():
    # Load pickle data object
    with open('sampling_deliverable.pickle', 'rb') as handle:
        data = pickle.load(handle)

    # Plot time vs value
    time_data = data.pop('time', None)
    values = []
    training_time = []
    for val in time_data:
        values.append(val[0])
        training_time.append(val[1])
    plt.title("Sampling vs Time")
    plt.xlabel("Sampling")
    plt.ylabel("Time")
    plt.plot(values, training_time)
    plt.savefig("sampling_time.png")
    plt.clf()

    for value, errors in data.items():
        num_of_trees = np.arange(len(errors))

        plt.xlabel("Number of Trees")
        plt.ylabel("RMSE")
        plt.title("Sampling Portion = {}".format(value))
        plt.plot(num_of_trees, errors)
        plt.legend(["Train", "Test"])
        plt.savefig("{}_sampling.png".format(value))
        plt.clf()


def plot_threshold_deliverable():
    # Load pickle data object
    with open('threshold_deliverable.pickle', 'rb') as handle:
        data = pickle.load(handle)

    # Plot time vs value
    time_data = data.pop('time', None)
    values = []
    training_time = []
    for val in time_data:
        values.append(val[0])
        training_time.append(val[1])
    plt.title("Threshold vs Time")
    plt.plot(values, training_time)
    plt.xlabel("Threshold")
    plt.ylabel("Time")
    plt.savefig("threshold_time.png")
    plt.clf()

    for value, errors in data.items():
        num_of_trees = np.arange(len(errors))

        plt.xlabel("Number of Trees")
        plt.ylabel("RMSE")
        plt.title("Sampling Portion = {}".format(value))
        plt.title("{}".format(value))
        plt.plot(num_of_trees, errors)
        plt.legend(["Train", "Test"])
        plt.savefig("threshold_{}.png".format(value))
        plt.clf()


def plot_deliverable():
    plot_depth_deliverable()
    plot_threshold_deliverable()
    plot_sampling_deliverable()

if __name__ == "__main__":
    plot_depth_deliverable()
    plot_threshold_deliverable()
    plot_sampling_deliverable()