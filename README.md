

---
# CS 6375 Machine Learning Assignment 1

## Overview
This repo contains my code for an assignment in CS 6375 Machine Learning. The main goal of this project is to train and compare the performance of feedforward and recurrent neural networks in the context of a sentiment analysis classification problem for Yelp reviews. I adjust their hidden dimension in order to analyze the hyperparameter’s effect on the models’ performance.

## Getting Started

### Preqreuisites
Need [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed

### Create and Activate the Conda Environment

Run the following command in the root directory of the project to create the conda environment:
```bash
conda env create -f environment.yml --name the_env_name
```
Activate the conda enviornment:
```bash
conda activate the_env_name
```

### Example Runs

**FFNN**
```bash
python rnn.py --hidden_dim 32 --epochs 10 --train_data new_training.json --val_data new_validation.json --test_data new_test.json
```
**RNN**
```bash
python rnn.py --hidden_dim 32 --epochs 10 --train_data new_training.json --val_data new_validation.json --test_data new_test.json
```

### Run Trials
Run the following command in the root directory of the project to run the trials for both models:
```bash
bash run.sh
```

