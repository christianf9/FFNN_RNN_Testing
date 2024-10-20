import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score
import pickle

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # obtain first hidden layer representation
        hidden_layer_rep = self.activation(self.W1(input_vector))
        # obtain output layer representation
        output_states = self.W2(hidden_layer_rep)
        # obtain probability dist.
        output_states = self.softmax(output_states)
        return output_states


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        test = json.load(test_f)

    tra = []
    val = []
    tst = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in test:
        tst.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val, tst


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", required=True, help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument("--trial", type=int, required=False, default=1, help="trial number")
    args = parser.parse_args()

    # fix random seeds
    random.seed(42 + args.trial - 1)
    torch.manual_seed(42 + args.trial - 1)

    # if GPU is available use it, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    os.makedirs(f"./vectorized_representations", exist_ok=True)
    vectorized_train_data_path = f"./vectorized_representations/vectorized_train_data.pkl"
    vectorized_valid_data_path = f"./vectorized_representations/vectorized_valid_data.pkl"
    vectorized_test_data_path = f"./vectorized_representations/vectorized_test_data.pkl"
    # check if vectorized data exists, if so load it, otherwise create it
    if os.path.exists(vectorized_train_data_path) and os.path.exists(vectorized_valid_data_path) and os.path.exists(vectorized_test_data_path):
        with open(vectorized_train_data_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(vectorized_valid_data_path, 'rb') as f:
            valid_data = pickle.load(f)
        with open(vectorized_test_data_path, 'rb') as f:
            test_data = pickle.load(f)
    else:
        print("========== Vectorizing data ==========")
        train_data = convert_to_vector_representation(train_data, word2index)
        valid_data = convert_to_vector_representation(valid_data, word2index)
        test_data = convert_to_vector_representation(test_data, word2index)
        with open(vectorized_train_data_path, 'wb') as f:
            pickle.dump(train_data, f)
        with open(vectorized_valid_data_path, 'wb') as f:
            pickle.dump(valid_data, f)
        with open(vectorized_test_data_path, 'wb') as f:
            pickle.dump(test_data, f)
    

    model = FFNN(input_dim = len(vocab), h = args.hidden_dim).to(device)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
    print("========== Training for {} epochs ==========".format(args.epochs))

    best_validation_accuracy = 0.0  # Track best validation accuracy
    os.makedirs(f"./saved_models/ffnn", exist_ok=True)
    best_model_path = f"./saved_models/ffnn/best_model_ffnn_{args.hidden_dim}_trial{args.trial}.pt"  # Path to save the best model
    os.makedirs(f"./logs", exist_ok=True)
    log_path = f"./logs/log_ffnn_{args.hidden_dim}.txt"
    with open(log_path, 'a') as log_f:
        log_f.write(f"Trial: {args.trial}\n")
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = None
            correct = 0
            total = 0
            start_time = time.time()
            loss_total = 0
            loss_count = 0
            print("Training started for epoch {}".format(epoch + 1))
            random.shuffle(train_data) # Good practice to shuffle order of training data
            minibatch_size = 16 
            N = len(train_data) 
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]

                    # move to respective device
                    input_vector = input_vector.to(device)
                    gold_label = torch.tensor([gold_label], device=device)

                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), gold_label)
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size
                loss_total += loss.data
                loss_count +=  1
                loss.backward()
                optimizer.step()
            print("Training completed for epoch {}".format(epoch + 1))
            print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            print("Training time for this epoch: {}".format(time.time() - start_time))

            model.eval()
            correct = 0
            total = 0
            start_time = time.time()
            print("Validation started for epoch {}".format(epoch + 1))
            minibatch_size = 16 
            N = len(valid_data) 
            for minibatch_index in tqdm(range(N // minibatch_size)):
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]

                    # move to respective device
                    input_vector = input_vector.to(device)
                    gold_label = torch.tensor([gold_label], device=device)

                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1

            validation_accuracy = correct / total
            trainning_accuracy = correct / total
            print("Validation completed for epoch {}".format(epoch + 1))
            print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            print("Validation time for this epoch: {}".format(time.time() - start_time))
            
            # Save the model if it's the best validation accuracy so far
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(model.state_dict(), best_model_path)

            loss_epoch = loss_total / loss_count
            log_f.write(f"Epoch: {epoch}\n")
            log_f.write(f"Loss: {loss_epoch}\n")
            log_f.write(f"Training accuracy: {trainning_accuracy}\n")
            log_f.write(f"Validation accuracy: {validation_accuracy}\n")


    print("========== Testing started ==========")
    # Load the best model based on validation accuracy
    model.load_state_dict(torch.load(best_model_path))
    # Set the model to evaluation mode
    model.eval()
    loss = None
    correct = 0
    total = 0
    minibatch_size = 16 
    N = len(test_data)
    # keep track of true and predicted labels
    y_true = []
    y_pred = []
    # iterate over test data
    for minibatch_index in tqdm(range(N // minibatch_size)):
        for example_index in range(minibatch_size):
            # get input vector and gold label
            input_vector, gold_label = test_data[minibatch_index * minibatch_size + example_index]
            # move to respective device
            input_vector = input_vector.to(device)
            gold_label = torch.tensor([gold_label], device=device)
            # get predicted vector and label
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector)
            # update correct and total
            correct += int(predicted_label == gold_label)
            total += 1
            # update true and predicted labels
            y_true.append(gold_label.item())
            y_pred.append(predicted_label.item())

    # calculate metrics
    testing_accuracy = correct / total
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')

    # print results
    print("Trial:", args.trial)
    print("Testing accuracy: {}".format(testing_accuracy))
    print("Macro F1 Score: {}".format(macro_f1))
    print("Macro Precision: {}".format(macro_precision))
    print("Macro Recall: {}".format(macro_recall))

    # save results
    os.makedirs(f"./results", exist_ok=True)
    results_path = f"./results/result_ffnn_{args.hidden_dim}.txt"
    with open(results_path, 'a') as results_f:
        results_f.write(f"Trial: {args.trial}\n")
        results_f.write(f"Testing accuracy: {testing_accuracy}\n")
        results_f.write(f"Macro F1 Score: {macro_f1}\n")
        results_f.write(f"Macro Precision: {macro_precision}\n")
        results_f.write(f"Macro Recall: {macro_recall}\n")