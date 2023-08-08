#!/bin/bash

### kitti
## 1 shot
# # train
# bash incremental_train.sh kitti 10 1 false false false 0 > logs/train_kitti_10_1_log.txt
# bash incremental_train.sh kitti 10 1 false false true 1 > logs/train_kitti_10_1_1_log.txt
# evaluation
bash incremental_evaluation.sh kitti 10 1 false false false 0 >logs/evaluation_kitti_10_1_log.txt
bash incremental_evaluation.sh kitti 10 1 false false true 1 > logs/evaluation_kitti_10_1_1_log.txt

# ## 5 shot
# # # train
# bash incremental_train.sh kitti 10 5 false false false 0 > logs/train_kitti_10_5_log.txt
# bash incremental_train.sh kitti 10 5 false false true 5 > logs/train_kitti_10_5_5_log.txt
# # evaluation
# bash incremental_evaluation.sh kitti 10 5 false false false 0 > logs/evaluation_kitti_10_5_log.txt
# bash incremental_evaluation.sh kitti 10 5 false false true 5 > logs/evaluation_kitti_10_5_5_log.txt


### cityscapes
## 1 shot
# train
# bash incremental_train.sh cityscapes 20 1 false true false 0 > logs/train_cityscapes_20_1_log.txt
# bash incremental_train.sh cityscapes 20 1 false true true 10 > logs/train_cityscapes_20_1_10_log.txt
# evaluation
# bash incremental_evaluation.sh cityscapes 20 1 false true false 0 > logs/evaluation_cityscapes_20_1_log.txt
# bash incremental_evaluation.sh cityscapes 20 1 false true true 10 > logs/evaluation_cityscapes_20_1_10_log.txt

## 5 shot
# train
# bash incremental_train.sh cityscapes 20 5 false true false 0 > logs/train_cityscapes_20_5_log.txt
# bash incremental_train.sh cityscapes 20 5 false true true 10 > logs/train_cityscapes_20_5_10_log.txt
# evaluation
# bash incremental_evaluation.sh cityscapes 20 5 false true false 0 > logs/evaluation_cityscapes_20_5_log.txt
# bash incremental_evaluation.sh cityscapes 20 5 false true true 10 > logs/evaluation_cityscapes_20_5_10_log.txt
