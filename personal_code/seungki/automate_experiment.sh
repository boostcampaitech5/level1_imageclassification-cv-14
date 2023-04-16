#!/bin/bash

# -- Define values for the loop
models=("EfficientnetB0" "EfficientnetB1" "EfficientnetB2" "EfficientnetB3")
learning_rates=(0.00002 0.00001 0.0001)
batch_sizes=(64 128)
epochs=20
augmentation="CustomAugmentation"
dataset="MaskSplitByProfileDataset"
valid_batch_size=256
criterion="cross_entropy"
optimizer="Adam"
lr_decay_step=5
resize_sizes=(224)
augmentation_types=["CenterCrop((320, 256))"]


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