#!/bin/bash

# -- Define values for the loop
models=("EfficientnetB0")
learning_rates=(0.00001 0.0001)
batch_sizes=(32)
epochs=35
# augmentation="CustomAugmentation"
# -- send augmentation type as argument to file name
# "centercrop" "randomerasing" "randomhorizontalflip" "colorjitter" "randomrotation" "gaussiannoise"
augmentation=("randomrotation")
dataset="MaskSplitByProfileDataset"
valid_batch_size=256
criterion=("focal")
optimizer="Adam"
lr_decay_step=5
augmentation_types="[look at augmentation name, experimenting one by one]"


# -- Loop over combination of values -- with augmentation loop
for model in "${models[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for augmentation_name in "${augmentation[@]}"; do
            # -- Run train.py of different values
                    python train.py --model "$model" --lr "$lr" --batch_size "$batch_size" --epochs "$epochs" --augmentation "$augmentation_name" --dataset "$dataset" --valid_batch_size "$valid_batch_size" --criterion "$criterion" --optimizer "$optimizer" --lr_decay_step "$lr_decay_step" --augmentation_types "$augmentation_types"
            done
        done
    done
done

# -- Loop over combination of values -- no augmentation loop
# for model in "${models[@]}"; do
#     for lr in "${learning_rates[@]}"; do
#         for batch_size in "${batch_sizes[@]}"; do
#             # -- Run train.py of different values
#                 python train.py --model "$model" --lr "$lr" --batch_size "$batch_size" --epochs "$epochs" --augmentation "$augmentation" --dataset "$dataset" --valid_batch_size "$valid_batch_size" --criterion "$criterion" --optimizer "$optimizer" --lr_decay_step "$lr_decay_step" --augmentation_types "$augmentation_types"
#         done
#     done
# done