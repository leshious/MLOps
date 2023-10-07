import pickle

import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split


def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=42
    )

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
