import json
import random

### FILE FOR CREATING NEW RANDOM SPLITS OF THE DATASET ###

train_size, valid_size, test_size = 16000, 800, 800

test_json_path = "Data_Embedding/test.json"
training_json_path = "Data_Embedding/training.json"
validation_json_path = "Data_Embedding/validation.json"

# load original data splits
with open(test_json_path, 'r') as f:
    test_data = json.load(f)
with open(training_json_path, 'r') as f:
    training_data = json.load(f)
with open(validation_json_path, 'r') as f:
    validation_data = json.load(f)

comb_data = test_data + training_data + validation_data # combine all the data
random.shuffle(comb_data) # shuffle the data

# split the data
new_training_data = comb_data[:16000]
new_validation_data = comb_data[16000:16000+800]
new_test_data = comb_data[16000+800:]

new_training_json_path = "new_training.json"
new_validation_json_path = "new_validation.json"
new_test_json_path = "new_test.json"

# write the new splits to new json files
with open(new_training_json_path, 'w') as f:
    json.dump(new_training_data, f)
with open(new_validation_json_path, 'w') as f:
    json.dump(new_validation_data, f)
with open(new_test_json_path, 'w') as f:
    json.dump(new_test_data, f)

