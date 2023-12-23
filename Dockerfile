# Используем базовый образ Triton Server от Nvidia без зависимостей от GPU
FROM nvcr.io/nvidia/tritonserver:23.12-py3

# Устанавливаем необходимые пакеты, например, ONNX Runtime
RUN pip install onnxruntime

# Копируем ваш ONNX-файл и скрипт деплоя в контейнер
COPY logistic_regression_model.onnx /models/logistic_regression_model/1/model.onnx
COPY triton_deploy.sh /models/logistic_regression_model/1/config.pbtxt

# Устанавливаем переменные окружения для Triton Server
ENV TRITON_MODEL_NAME=logistic_regression_model
ENV TRITON_MODEL_PATH=/models/logistic_regression_model

# Запускаем Triton Server
CMD ["tritonserver", "--model-repository=/models"]
