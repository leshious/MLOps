name: "logistic_regression_model"
backend: "onnxruntime"
max_batch_size: 1
input [
  {
    name: "float_input"
    data_type: TYPE_FP32
    dims: [ -1, 3 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ -1, 3 ]
  }
]
