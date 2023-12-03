import pickle

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import hydra
import os

@hydra.main(config_path="configs", config_name="logistic_regression_config", version_base="1.2")
def main(cfg):
    working_dir = hydra.utils.get_original_cwd()

    # Соединяем путь к файлам данных
    data_path = f"{working_dir}/{cfg.data.data_dir}/"

    # Загружаем данные
    X_train = np.load(data_path + 'X_train.npy')
    X_test = np.load(data_path + 'X_test.npy')
    y_train = np.load(data_path + 'y_train.npy')
    y_test = np.load(data_path + 'y_test.npy')


    # Initialize and train the logistic regression model
    clf = LogisticRegression(
        max_iter=cfg.max_iter,
        penalty=cfg.penalty,
        fit_intercept=cfg.fit_intercept,
        C=1.0,
        multi_class=cfg.multi_class,
    ).fit(X_train, y_train)

    with open("logistic_regression_model.pkl", "wb") as file:
        pickle.dump(clf, file)


if __name__ == "__main__":
    main()
