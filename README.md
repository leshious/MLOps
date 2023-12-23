# MLOps

## Introduction
This repository contains the first homework assignment for the MLOps course. The project focuses on demonstrating the workflow for running machine learning operations efficiently.

## Что реализовал: 
1. DVC - смог, все работает
2. Hydra - все круто, все работает
3. Logging - не завелось(((
4. Inference - все работает, все круто 

## Installation
To run the code, follow these steps:

1. Create a Conda environment:
    ```bash
    conda create --name mlopsik python=3.9
    ```

2. Activate the Conda environment:
    ```bash
    conda activate mlopsik
    ```

3. Install project dependencies using Poetry:
    ```bash
    poetry install
    ```

4. Pull the necessary data using DVC:
    ```bash
    dvc pull
    ```

## Usage
Once the setup is complete, run the following commands in sequence:

1. Train the model:
    ```bash
    poetry run python train.py
    ```

2. Start the server:
    ```bash
    poetry run python run_server.py
    ```

3. Find the results in the `inference` directory.



# ДЗ 3 или боль и страдания с склерн. 

## Предыстория. 
Я решил пойти не самым умным путем и в качестве модели выбрал sklearn.logistic_regression. Обучил ее на ирис. Все работало и даже экспортировалось в ONNX, единственное, что референсов таких работ на гитхабе и примеров таких запусков найти было тяжело. Но я справился. Это была дз2. 

Наступила ДЗ3.

## Мой путь боли. 

Я решил и дальше использовать свою модель в формате ONNX. Ничего не предвещало беды. Поскольку эта часть дз делалась за не так много времени, я решил оставить ее на потом (спойлер: тупое решение). 

Я завел докер образ. Дальше я решил поднять в нем модель и получил такую ошибку:

| logistic_regression_model | 1       | UNAVAILABLE: Unsupported: Unsupported ONNX Type 'ONNX_TYPE_SEQUENCE' for I/O 'output_probability', expected 'ONNX_TYPE_TENSOR'. |


Я пытался ее поправить, сделал все мыслимые и немыслемые исправления кода. Но потом полез гуглить. Оказалось, что эта проблема на стороне ONNX и решить я ее сам не могу) https://github.com/triton-inference-server/onnxruntime_backend/issues/94#issuecomment-1130317325

Такие дела) Было бы больше времени у меня, я бы попробовал переписать модель на торче и сделать человеческим путем, но я дурак)

P.S. надеюсь меня за это не отчислят. 







## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the [MIT License](LICENSE).
