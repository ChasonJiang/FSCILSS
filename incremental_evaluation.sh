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
## Evaluation
## If you are working with environments make sure you activated it before starting the script

# Stage 1, first Teacher
if test $enable_stage_1 = "true"
then
    python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set1 --train_set 1 --weights_epoch $teacher_num_epoch --task_to_val 1 --eval_base_model --save_evaluation_results
fi

# Stage 2, with first teacher
if test $enable_pl = "true"
then
    python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch $teacher_num_epoch --task_to_val 1 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --pseudo_label_mode --num_pseudo_labels $num_pseudo_labels --num_shot $num_shot
    python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch $teacher_num_epoch --task_to_val 2 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --pseudo_label_mode --num_pseudo_labels $num_pseudo_labels --num_shot $num_shot
    python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch $teacher_num_epoch --task_to_val 12 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --pseudo_label_mode --num_pseudo_labels $num_pseudo_labels --num_shot $num_shot
else
    python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch $teacher_num_epoch --task_to_val 1 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --num_shot $num_shot
    python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch $teacher_num_epoch --task_to_val 2 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --num_shot $num_shot
    python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch $teacher_num_epoch --task_to_val 12 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --num_shot $num_shot
fi

# Stage 3, with second teacher
if test $enable_stage_3 = "true"
then
    if test $enable_pl = "true"
    then
        python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch $teacher_num_epoch --task_to_val 1 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --pseudo_label_mode --num_pseudo_labels $num_pseudo_labels --num_shot $num_shot
        python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch $teacher_num_epoch --task_to_val 2 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --pseudo_label_mode --num_pseudo_labels $num_pseudo_labels --num_shot $num_shot
        python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch $teacher_num_epoch --task_to_val 3 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --pseudo_label_mode --num_pseudo_labels $num_pseudo_labels --num_shot $num_shot
        python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch $teacher_num_epoch --task_to_val 12 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --pseudo_label_mode --num_pseudo_labels $num_pseudo_labels --num_shot $num_shot
        python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch $teacher_num_epoch --task_to_val 123 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --pseudo_label_mode --num_pseudo_labels $num_pseudo_labels --num_shot $num_shot
    else
        python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch $teacher_num_epoch --task_to_val 1 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --num_shot $num_shot
        python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch $teacher_num_epoch --task_to_val 2 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --num_shot $num_shot
        python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch $teacher_num_epoch --task_to_val 3 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --num_shot $num_shot
        python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch $teacher_num_epoch --task_to_val 12 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --num_shot $num_shot
        python3 evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch $teacher_num_epoch --task_to_val 123 --multi_exp $num_exp --save_evaluation_results --dataset $dataset --num_shot $num_shot
    fi
fi

### Static
## Evaluation
# All classes
# python3 evaluate_erfnet.py --load_model_name erfnet_static_set123 --train_set 123 --weights_epoch $teacher_num_epoch --task_to_val 1
# python3 evaluate_erfnet.py --load_model_name erfnet_static_set123 --train_set 123 --weights_epoch $teacher_num_epoch --task_to_val 2
# python3 evaluate_erfnet.py --load_model_name erfnet_static_set123 --train_set 123 --weights_epoch $teacher_num_epoch --task_to_val 12
# python3 evaluate_erfnet.py --load_model_name erfnet_static_set123 --train_set 123 --weights_epoch $teacher_num_epoch --task_to_val 3
# python3 evaluate_erfnet.py --load_model_name erfnet_static_set123 --train_set 123 --weights_epoch $teacher_num_epoch --task_to_val 123

