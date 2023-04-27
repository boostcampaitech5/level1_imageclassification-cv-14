#!/bin/bash

# -- Define values for the loop
models=("EfficientnetB0")
learning_rates=(0.0001)
batch_sizes=(64)
epochs=25
# augmentation="CustomAugmentation"
# -- send augmentation type as argument to file name
# "centercrop" "randomerasing" "randomhorizontalflip" "colorjitter" "randomrotation" "gaussiannoise"
# augmentation=("CustomAugmentation")
augmentation=("BaseAugmentation")
dataset="MaskSplitByProfileDataset"
valid_batch_size=256
# criterion="focal"
# -- for hyperparameter loop
# "labelsmoothing" "F1"
criterion=("focal")
optimizer="Adam"
lr_decay_step=5
# weight_decay=(0.0001 0.001)
# -- for hyperparameter loop
# lr_decay_step=(5 3)
augmentation_types=""


# -- Loop over combination of values -- with augmentation loop
for model in "${models[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for augmentation_name in "${augmentation[@]}"; do
                for cr in "${criterion[@]}"; do
                        # -- Run train.py of different values
                        python train.py --model "$model" --lr "$lr" --batch_size "$batch_size" --epochs "$epochs" --augmentation "$augmentation_name" --dataset "$dataset" --valid_batch_size "$valid_batch_size" --criterion "$cr" --optimizer "$optimizer" --lr_decay_step "$lr_decay_step" --augmentation_types "$augmentation_types"
                done
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

# -- Loop over hyperparameter
# for model in "${models[@]}"; do
#     for lr in "${learning_rates[@]}"; do
#         for batch_size in "${batch_sizes[@]}"; do
#             for cr in "${criterion[@]}"; do
#                 for lds in "${lr_decay_step[@]}"; do
#                     for op in "${optimizer[@]}"; do
#                         # -- Run train.py of different values
#                             python train.py --model "$model" --lr "$lr" --batch_size "$batch_size" --epochs "$epochs" --augmentation "$augmentation" --dataset "$dataset" --valid_batch_size "$valid_batch_size" --criterion "$cr" --optimizer "$op" --lr_decay_step "$lds" --augmentation_types "$augmentation_types"
#                     done
#                 done
#             done
#         done
#     done
# done

# for model in "${models[@]}"; do
#     for lr in "${learning_rates[@]}"; do
#         for batch_size in "${batch_sizes[@]}"; do
#             for augmentation_name in "${augmentation[@]}"; do
#                 for wd in "${weight_decay[@]}"; do
#             # -- Run train.py of different values
#                         python train.py --model "$model" --lr "$lr" --batch_size "$batch_size" --epochs "$epochs" --augmentation "$augmentation_name" --dataset "$dataset" --valid_batch_size "$valid_batch_size" --criterion "$criterion" --optimizer "$optimizer" --lr_decay_step "$lr_decay_step" --augmentation_types "$augmentation_types" --weight_decay "$wd"
#                 done
#             done
#         done
#     done
# done