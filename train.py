import pickle

import hydra
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression


@hydra.main(config_path="configs", config_name="lr_config", version_base="1.2")
def main(cfg):
    working_dir = hydra.utils.get_original_cwd()

    data_path = f"{working_dir}/{cfg.data.data_dir}/"

    X_train = np.load(data_path + "X_train.npy")
    # X_test = np.load(data_path + "X_test.npy")
    y_train = np.load(data_path + "y_train.npy")
    # y_test = np.load(data_path + "y_test.npy")

    clf = LogisticRegression(
        max_iter=cfg.max_iter,
        penalty=cfg.penalty,
        fit_intercept=cfg.fit_intercept,
        C=1.0,
        multi_class=cfg.multi_class,
    ).fit(X_train, y_train)

    with open("logistic_regression_model.pkl", "wb") as file:
        pickle.dump(clf, file)

    initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type)
    with open("logistic_regression_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())


if __name__ == "__main__":
    main()
