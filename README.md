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

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the [MIT License](LICENSE).
