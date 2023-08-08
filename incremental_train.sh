#!/bin/bash
dataset=$1 # string "cityscapes" or "kitti"
num_exp=$2
num_shot=$3
num_epoch=200
teacher_num_epoch=`expr $num_epoch - 1`
enable_stage_1=$4 # string "true" or "false"
enable_stage_3=$5 # string "true" or "false"
enable_pl=$6 # string "true" or "false"
num_pseudo_labels=$7



### Incremental
## Train
## If you are working with environments make sure you activated it before starting the script

# Stage 1, first Teacher
if test $enable_stage_1 = "true"
then
    python3 train_erfnet_static.py --model_name erfnet_incremental_set1 --train_set 1 --num_epochs $num_epoch --city
fi

# Stage 2, with first teacher
if test $enable_pl = "true"
then
    python3 train_erfnet_incremental.py --model_name erfnet_incremental_set12 --train_set 2 --num_epochs $num_epoch --teachers erfnet_incremental_set1 $teacher_num_epoch 1 --num_exp $num_exp --num_shot $num_shot --pseudo_label_mode --num_pseudo_labels $num_pseudo_labels --dataset $dataset
else
    python3 train_erfnet_incremental.py --model_name erfnet_incremental_set12 --train_set 2 --num_epochs $num_epoch --teachers erfnet_incremental_set1 $teacher_num_epoch 1 --num_exp $num_exp --num_shot $num_shot --dataset $dataset
fi

# Stage 3, with second teacher
if test $enable_stage_3 = "true"
then
    if test $enable_pl = "true"
    then
        python3 train_erfnet_incremental.py --model_name erfnet_incremental_set123 --train_set 3 --num_epochs $num_epoch --teachers erfnet_incremental_set12 $teacher_num_epoch 2 --num_exp $num_exp --num_shot $num_shot --load_best_model --pseudo_label_mode --num_pseudo_labels $num_pseudo_labels --dataset $dataset
    else
        python3 train_erfnet_incremental.py --model_name erfnet_incremental_set123 --train_set 3 --num_epochs $num_epoch --teachers erfnet_incremental_set12 $teacher_num_epoch 2 --num_exp $num_exp --num_shot $num_shot --load_best_model --dataset $dataset
    fi
fi

### Static
## Train
# All classes baseline
# python3 train_erfnet_static.py --model_name erfnet_static_set123 --train_set 123 --num_epochs $num_epoch
