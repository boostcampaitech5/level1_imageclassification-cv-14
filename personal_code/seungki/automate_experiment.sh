#!/bin/bash

# -- Define values for the loop
models=("model1" "model2" "model3")
learning_rates=(0.001 0.01 0.1)
batch_sizes=(32 64 128)
epochs=20
augmentation="BaseAugmentation"
dataset="MaskSplitByProfileDataset"
valid_batch_size=256
criterion="cross_entropy"
optimizer="Adam"
lr_decay_step=5
resize_sizes=(224)
augmentation_types=[]


# -- Loop over combination of values
for model in "${models[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for resize in "${resize_sizes[@]}"; do
                # -- Run train.py of different values
                python train.py --model "$model" --lr "$lr" --batch_size "$batch_size" --epochs "$epochs" --augmentation "$augmentation" --dataset "$dataset" --valid_batch_size "$valid_batch_size" --criterion "$criterion" --optimizer "$optimizer" --lr_decay_step "$lr_decay_step" --resize "$resize" --augmentation_types "$augmentation_types"
            done
        done
    done
done