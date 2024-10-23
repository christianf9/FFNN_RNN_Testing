# CS 6375 Machine Learning Assignment 1

## Overview
This repo contains my code for an assignment in CS 6375 Machine Learning. The main goal of this project is to train and compare the performance of feedforward and recurrent neural networks in the context of a sentiment analysis classification problem for Yelp reviews. I adjust their hidden dimension in order to analyze the hyperparameter’s effect on the models’ performance.

## Important Files/Directories
### Root Directory:
- **ffnn.py**: Python script for Feedforward Neural Network (FFNN)
- **rnn.py**: Python script for Recurrent Neural Network (RNN)
- **run.sh**: Shell script for running trials for both FFNN and RNN
- **new_data_splits/**: Contains the new data splits that I used for testing
- **my_logs/**: Contains the log files for my trial runs
- **my_results/**: Contains the result files for my trial runs
- **environment.yml**: YAML file to setup conda enviornment that I used
- **requirements.txt**: Requirements file that lists the Python dependencies of the project
- **Data_Embedding.zip**: Contains word embeddings for use with the RNN models

### Utils Directory:
- **split_data.py**: Python script to improve old data splits for a better distribution of classes
- **coverage_analysis_bow.py**: Python script to calculate the coverage of the vocab from the training data for the validation and testing splits for the bag of word approach
- **coverage_analysis_pretrained.py**: Python script to calculate the coverage of the vocab from the training data for the validation and testing splits for the pretrained word embeddings approach

## Results and Logs
My results and logs for my trial runs can be seen in my_results and my_logs folders, respectively. To evaluate both the feedforward neural networks and the recurrent neural networks, I varied the hidden dimension through five different hidden dimension sizes: 16, 32, 64, 128, and 256. I trained each model for five trials, with 10 epochs per trial, and a mini-batch size of 16. For each trial, I set the random seed to 41 plus the trial number starting at trial 1 for Python’s built-in random number generator and PyTorch’s random number generator. This ensures reproducibility and varies the weight initializations of the models and shuffling of the training split to eliminate variability due to randomness and get more reliable performance estimates.

## Getting Started

***FIRST EXTRACT "Data_Embedding.zip"*** so the RNN model has access to the word embeddings. Make sure that after extraction there is a file structure "./Data_Embedding/word_embedding.pkl".

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
python ffnn.py --hidden_dim 32 --epochs 10 --train_data new_data_splits/new_training.json --val_data new_data_splits/new_validation.json --test_data new_data_splits/new_test.json
```
**RNN**
```bash
python rnn.py --hidden_dim 32 --epochs 10 --train_data new_data_splits/new_training.json --val_data new_data_splits/new_validation.json --test_data new_data_splits/new_test.json
```

### Run Trials
Run the following command in the root directory of the project to run the trials for both models:
```bash
bash run.sh
```
The logs during training will be stored in a file "./logs/log_modelName_hiddenDim.txt" formatted as:
```bash
Trial: 1
Epoch: 0
Loss: 1.3931065797805786
Training accuracy: 0.48625
Validation accuracy: 0.48625
Epoch: 1
Loss: 1.1995344161987305
Training accuracy: 0.4825
Validation accuracy: 0.4825
Epoch: 2
Loss: 1.1451973915100098
Training accuracy: 0.50125
Validation accuracy: 0.50125
...
```
The results after training will be stored in a file "./results/result_modelName_hiddenDim.txt" formated as:
```bash
Trial: 1
Testing accuracy: 0.54125
Macro F1 Score: 0.5348826602279703
Macro Precision: 0.5350758918500853
Macro Recall: 0.5394086349213603
Trial: 2
Testing accuracy: 0.51625
Macro F1 Score: 0.51252144415715
Macro Precision: 0.518847441579226
Macro Recall: 0.5161641311362265
Trial: 3
Testing accuracy: 0.52625
Macro F1 Score: 0.5320839717561386
Macro Precision: 0.5501887389320821
Macro Recall: 0.5265503017043895
...
```
