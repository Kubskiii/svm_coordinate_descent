#!/usr/local/bin/python
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="Directory with results CSV")
    parser.add_argument("-f", "--format", default="png", help="Output format of generated plots")
    args = parser.parse_args()
    directory = args.directory
    format = args.format
    data_dict = dict()

    for file in os.listdir(directory):
        filepath = os.path.join(directory, file)
        if os.path.isfile(filepath):
            file_base, file_ext = os.path.splitext(file)
            if file_ext == ".csv":
                r = re.search(r'(?P<model>.*)_(?P<dataset>\d+_\d+)$', file_base)
                d = data_dict.get(r.group("dataset"), dict())
                d[r.group("model")] = pd.read_csv(filepath)
                data_dict[r.group("dataset")] = d

    plots_directory = os.path.join(directory, "plots")
    os.mkdir(plots_directory)
    plt.style.use("ggplot")
    for dataset, dataset_dict in data_dict.items():
        for y in ["loss", "loss_prime", "train_acc", "test_acc"]:
            for model, data in dataset_dict.items():
                plt.plot(range(1, len(data) + 1), data[y], label=model)
            plt.legend()
            plt.xlabel("iteration")
            plt.ylabel(y)
            plt.savefig(os.path.join(plots_directory, f"{dataset}_{y}.{format}"))
            plt.clf()


if __name__ == "__main__":
    main()
