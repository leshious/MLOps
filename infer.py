import pickle

import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')

    with open("logistic_regression_model.pkl", "rb") as file:
        loaded_model = pickle.load(file)

    y_test_pred = loaded_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(test_accuracy)
    print(
        precision_score(y_test, y_test_pred, average="macro")
    )  # added average='macro' for multi-class case
    print(
        f1_score(y_test, y_test_pred, average="macro")
    )  # added average='macro' for multi-class case

    results_df = pd.DataFrame({"Predictions": y_test_pred})

    # Save the dataframe to a .csv file
    results_df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()
