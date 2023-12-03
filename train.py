import pickle

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')

    # Initialize and train the logistic regression model
    clf = LogisticRegression(max_iter=1000, penalty='l2', fit_intercept=True).fit(X_train, y_train)

    with open("logistic_regression_model.pkl", "wb") as file:
        pickle.dump(clf, file)


if __name__ == "__main__":
    main()
