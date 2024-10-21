# CS 6375 Machine Learning Assignment 1

## Overview
This repo contains my code for an assignment in CS 6375 Machine Learning. The main goal of this project is to train and compare the performance of feedforward and recurrent neural networks in the context of a sentiment analysis classification problem for Yelp reviews. I adjust their hidden dimension in order to analyze the hyperparameter’s effect on the models’ performance.

## Results and Logs
My results and logs for my trial runs can be seen in my_results and my_logs folders, respectively. To evaluate both the feedforward neural networks and the recurrent neural networks, I varied the hidden dimension through five different hidden dimension sizes: 16, 32, 64, 128, and 256. I trained each model for five trials, with 10 epochs per trial, and a mini-batch size of 16. For each trial, I set the random seed to 41 plus the trial number starting at trial 1 for PYthon’s built-in random number generator and PyTorch’s random number generator. This ensures reproducibility and varies the weight initializations of the models and shuffling of the training split to eliminate variability due to randomness and get more reliable performance estimates.

## Getting Started

### Preqreuisites
Need [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed

### Create and Activate the Conda Environment

#### For CPU Usage Only:
```bash
conda create --name the_env_name python=3.8
conda activate the_env_name
pip install -r requirements.txt
```

#### Environment Used for Testing on a GPU with CUDA Version 12.6:
```bash
conda env create -f environment.yml --name the_env_name
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

