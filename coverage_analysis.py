import json
import string

### CODE FOR CALCULATING COVERAGE OF THE VOCABULARY FOR VALIDATION AND TESTING SPLITS ###

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

# make the vocab
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab

# calculate the coverage
def calc_coverage(data, vocab):
    not_found = 0 # number of words not found in the vocab
    total = 0 # total number of words
    # iterate over the data and count the number of words not found in the vocab
    for document, _ in data:
        for word in document:
            total += 1
            if word not in vocab:
                not_found += 1
    return ((total-not_found) / total) * 100

# make the vocab from the training data
vocab = make_vocab(tra)

# print the coverage
print(f"Validation data coverage: {calc_coverage(val, vocab):.3f}%")
print(f"Test data coverage: {calc_coverage(tst, vocab):.3f}%")

