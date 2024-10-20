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
import string
from argparse import ArgumentParser
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):  
        # obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        output, hidden = self.rnn(inputs)
        # obtain output layer representations
        output_layer_rep = self.W(output)
        # sum over output
        output_sum = output_layer_rep.sum(dim=0)
        # obtain probability dist.
        predicted_vector = self.softmax(output_sum)
        return predicted_vector


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
    parser.add_argument("--trial", type=int, required=True, help="trial number")
    args = parser.parse_args()

    # fix random seeds
    random.seed(42 + args.trial - 1)
    torch.manual_seed(42 + args.trial - 1)

    # if GPU is available use it, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim).to(device)  # Fill in parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    word_embedding = pickle.load(open('./Data_Embedding/word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

    last_validation_accuracy = 0.0
    best_validation_accuracy = 0.0

    os.makedirs(f"./saved_models/rnn", exist_ok=True)
    best_model_path = f"./saved_models/rnn/best_model_rnn_{args.hidden_dim}_trial{args.trial}.pt"  # Path to save the best model
    os.makedirs(f"./logs", exist_ok=True)
    log_path = f"./logs/log_rnn_{args.hidden_dim}.txt"
    with open(log_path, 'a') as log_f:
        log_f.write(f"Trial: {args.trial}\n")
        for epoch in range(args.epochs):
            random.shuffle(train_data)
            model.train()
            # You will need further code to operationalize training, ffnn.py may be helpful
            print("Training started for epoch {}".format(epoch + 1))
            train_data = train_data
            correct = 0
            total = 0
            minibatch_size = 16
            N = len(train_data)

            loss_total = 0
            loss_count = 0
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                    gold_label = torch.tensor([gold_label], device=device)
                    input_words = " ".join(input_words)

                    # Remove punctuation
                    input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                    # Look up word embedding dictionary
                    vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]

                    # Transform the input into required shape
                    vectors = np.array(vectors)
                    vectors = torch.tensor(vectors).view(len(vectors), 1, -1).to(device)
                    output = model(vectors)

                    # Get loss
                    example_loss = model.compute_Loss(output.view(1,-1), gold_label)

                    # Get predicted label
                    predicted_label = torch.argmax(output)

                    correct += int(predicted_label == gold_label)
                    # print(predicted_label, gold_label)
                    total += 1
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss

                loss = loss / minibatch_size
                loss_total += loss.data
                loss_count += 1
                loss.backward()
                optimizer.step()
            print(loss_total/loss_count)
            print("Training completed for epoch {}".format(epoch + 1))
            print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            trainning_accuracy = correct/total


            model.eval()
            correct = 0
            total = 0
            print("Validation started for epoch {}".format(epoch + 1))
            valid_data = valid_data

            for input_words, gold_label in tqdm(valid_data):
                gold_label = torch.tensor([gold_label], device=device)
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                        in input_words]
                vectors = np.array(vectors)
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1).to(device)
                output = model(vectors)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
                # print(predicted_label, gold_label)
            print("Validation completed for epoch {}".format(epoch + 1))
            print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))

            validation_accuracy = correct/total

            # Save the model if it's the best validation accuracy so far
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(model.state_dict(), best_model_path)

            epoch += 1

            loss_epoch = loss_total/loss_count
            log_f.write(f"Epoch: {epoch}\n")
            log_f.write(f"Loss: {loss_epoch}\n")
            log_f.write(f"Training accuracy: {trainning_accuracy}\n")
            log_f.write(f"Validation accuracy: {validation_accuracy}\n")

    # Testing phase
    print("========== Testing started ==========")
    # Load the best model based on validation accuracy
    model.load_state_dict(torch.load(best_model_path))
    # Set the model to evaluation mode
    model.eval()
    correct = 0
    total = 0
    # Keep track of true and predicted labels
    y_true = []
    y_pred = []
    # Iterate over the test data
    for input_words, gold_label in tqdm(test_data):
        # Convert the gold label to a tensor
        gold_label = torch.tensor([gold_label], device=device)
        # Convert list of words to a string
        input_words = " ".join(input_words)
        # Remove punctuation and split into words
        input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
        # Convert words to vectors with the predefined word embeddings
        vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                    in input_words]
        vectors = np.array(vectors)
        vectors = torch.tensor(vectors).view(len(vectors), 1, -1).to(device)
        # Get the model's prediction
        output = model(vectors)
        # update correct and total
        predicted_label = torch.argmax(output)
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

    # Print the results
    print("Trial:", args.trial)
    print("Testing accuracy: {}".format(testing_accuracy))
    print("Macro F1 Score: {}".format(macro_f1))
    print("Macro Precision: {}".format(macro_precision))
    print("Macro Recall: {}".format(macro_recall))

    # Save the results
    os.makedirs(f"./results", exist_ok=True)
    results_path = f"./results/result_rnn_{args.hidden_dim}.txt"
    with open(results_path, 'a') as results_f:
        results_f.write(f"Trial: {args.trial}\n")
        results_f.write(f"Testing accuracy: {testing_accuracy}\n")
        results_f.write(f"Macro F1 Score: {macro_f1}\n")
        results_f.write(f"Macro Precision: {macro_precision}\n")
        results_f.write(f"Macro Recall: {macro_recall}\n")
    
    # You may find it beneficial to keep track of training accuracy or training loss;

    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
