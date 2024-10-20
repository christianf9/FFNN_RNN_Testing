#!/bin/bash

# Model names to test
model_names=("rnn","ffnn")

# Hidden dimensions to test
hidden_dims=(16 32 64 128 256)

# Dataset split paths
train_data="new_data_splits/new_training.json"
val_data="new_data_splits/new_validation.json"
test_data="new_data_splits/new_test.json"

# Epochs for training
epochs=10

for model_name in "${model_names[@]}"
    for trial in {1,2,3,4,5} # trial number determines the random seed (41 + trial #)
    do
        for hidden_dim in "${hidden_dims[@]}"
        do
            echo "Trial $trial, Hidden dimension = $hidden_dim, model = $model_name"
            
            if [ "$model_name" = "ffnn" ]; then
                python ffnn.py \
                    --hidden_dim $hidden_dim \
                    --epochs $epochs \
                    --train_data $train_data \
                    --val_data $val_data \
                    --test_data $test_data \
                    --trial $trial
            elif [ "$model_name" = "rnn" ]; then
                python rnn.py \
                    --hidden_dim $hidden_dim \
                    --epochs $epochs \
                    --train_data $train_data \
                    --val_data $val_data \
                    --test_data $test_data \
                    --trial $trial
            else
                echo "Error: unknown model \"$model_name\""
            fi
        done
    done
done
