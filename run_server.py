import os

import numpy as np
import onnxruntime as ort
import pandas as pd


def run_server():
    onnx_model = "logistic_regression_model.onnx"
    ort_session = ort.InferenceSession(onnx_model)

    input_data = np.load("data/X_test.npy").astype(np.float32)

    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    result = ort_session.run([output_name], {input_name: input_data})

    if not os.path.exists("inference"):
        os.makedirs("inference")

    output_array = np.array(result).reshape(-1, 1)
    output_df = pd.DataFrame(output_array, columns=["prediction"])
    output_df.to_csv("inference/output.csv", index=False)


if __name__ == "__main__":
    run_server()
