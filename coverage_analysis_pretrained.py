import json
import string
import pickle

### CODE FOR CALCULATING COVERAGE OF THE VOCABULARY FOR VALIDATION AND TESTING SPLITS FOR PRETRAINED WORD EMBEDDINGS ###

# data splits' paths
training_data_path = "new_data_splits/new_training.json"
validation_data_path = "new_data_splits/new_validation.json"
test_data_path = "new_data_splits/new_test.json"

# load data splits
with open(training_data_path, 'r') as f:
    training_data = json.load(f)
with open(validation_data_path, 'r') as f:
    validation_data = json.load(f)
with open(test_data_path, 'r') as f:
    test_data = json.load(f)

# load the word embedding
word_embedding_path = "Data_Embedding/word_embedding.pkl"
with open(word_embedding_path, 'rb') as f:
    word_embedding = pickle.load(f)

# preprocess data
tra = []
val = []
tst = []
for elt in training_data:
    tra.append((elt["text"].translate(str.maketrans('', '', string.punctuation)).split(),int(elt["stars"]-1)))
for elt in validation_data:
    val.append((elt["text"].translate(str.maketrans('', '', string.punctuation)).split(),int(elt["stars"]-1)))
for elt in test_data:
    tst.append((elt["text"].translate(str.maketrans('', '', string.punctuation)).split(),int(elt["stars"]-1)))

# check how many words in training set are not in the word embedding
not_found = 0
total = 0
for input_words, gold_label in tra:
    input_words = " ".join(input_words)
    input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
    for word in input_words:
        total += 1
        if word not in word_embedding.keys():
            not_found += 1

# check how many words in validation set are not in the word embedding
not_found_val = 0
total_val = 0
for input_words, gold_label in val:
    input_words = " ".join(input_words)
    input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
    for word in input_words:
        total_val += 1
        if word not in word_embedding.keys():
            not_found_val += 1

# check how many words in test set are not in the word embedding
not_found_test = 0
total_test = 0
for input_words, gold_label in tst:
    input_words = " ".join(input_words)
    input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
    for word in input_words:
        total_test += 1
        if word not in word_embedding.keys():
            not_found_test += 1
            
# print the percentage of words found in the word embedding
print(f"Training data coverage: {(1 - not_found/total) * 100:.3f}%")
print(f"Validation data coverage: {(1 - not_found_val/total_val) * 100:.3f}%")
print(f"Test data coverage: {(1 - not_found_test/total_test) * 100:.3f}%")