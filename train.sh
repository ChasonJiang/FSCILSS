#!/bin/bash

# base model using cityscapes

python3 train_erfnet_static.py --model_name erfnet_incremental_set1 --train_set 1 --num_epochs 199 --city > logs/train_base_model.log

# kitti
## 1 shot
### without pseudo labels
bash incremental_train.sh kitti 10 1 false false false 0 > logs/train_kitti_10_1.log
### with pseudo labels
bash incremental_train.sh kitti 10 1 false false true 1 > logs/train_kitti_10_1_1.log

## 5 shot
bash incremental_train.sh kitti 10 5 false false false 0 > logs/train_kitti_10_5.log
bash incremental_train.sh kitti 10 5 false false true 5 > logs/train_kitti_10_5_5.log

# cityscapes
## 1 shot
bash incremental_train.sh cityscapes 20 1 false true false 0 > logs/train_cityscapes_20_1.log
bash incremental_train.sh cityscapes 20 1 false true true 10 > logs/train_cityscapes_20_1_10.log

## 5 shot
bash incremental_train.sh cityscapes 20 5 false true false 0 > logs/train_cityscapes_20_5.log
bash incremental_train.sh cityscapes 20 5 false true true 10 > logs/train_cityscapes_20_5_10.log
