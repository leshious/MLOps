import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score


def main():
    X_train = np.load("data/X_train.npy")
    X_test = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")

    with open("logistic_regression_model.pkl", "rb") as file:
        loaded_model = pickle.load(file)

    y_test_pred = loaded_model.predict(X_test)
    y_train_pred = loaded_model.predict(X_train)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(test_accuracy)
    print(precision_score(y_test, y_test_pred, average="macro"))
    print(f1_score(y_test, y_test_pred, average="macro"))
    print(train_accuracy)

    results_df = pd.DataFrame({"Predictions": y_test_pred})

    results_df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()
