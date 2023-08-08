#!/bin/bash

# kitti
## 1 shot
### without pseudo labels
bash incremental_evaluation.sh kitti 10 1 false false false 0 >logs/evaluation_kitti_10_1.log
### with pseudo labels
bash incremental_evaluation.sh kitti 10 1 false false true 1 > logs/evaluation_kitti_10_1_1.log

## 5 shot
bash incremental_evaluation.sh kitti 10 5 false false false 0 > logs/evaluation_kitti_10_5.log
bash incremental_evaluation.sh kitti 10 5 false false true 5 > logs/evaluation_kitti_10_5_5.log


# cityscapes
## 1 shot
bash incremental_evaluation.sh cityscapes 20 1 false true false 0 > logs/evaluation_cityscapes_20_1.log
bash incremental_evaluation.sh cityscapes 20 1 false true true 10 > logs/evaluation_cityscapes_20_1_10.log

## 5 shot
bash incremental_evaluation.sh cityscapes 20 5 false true false 0 > logs/evaluation_cityscapes_20_5.log
bash incremental_evaluation.sh cityscapes 20 5 false true true 10 > logs/evaluation_cityscapes_20_5_10.log
