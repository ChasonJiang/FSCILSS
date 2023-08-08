@echo off
setlocal

set dataset=%1
set num_exp=%2
set num_shot=%3
set num_epoch=200
set /a teacher_num_epoch=%num_epoch% - 1
set enable_stage_1=%4
set enable_stage_3=%5
set enable_pl=%6
set num_pseudo_labels=%7

REM Incremental
REM Train

REM Stage 1, first Teacher
if "%enable_stage_1%"=="true" (
    python train_erfnet_static.py --model_name erfnet_incremental_set1 --train_set 1 --num_epochs %num_epoch% --city
)

REM Stage 2, with first teacher
if "%enable_pl%"=="true" (
    python train_erfnet_incremental.py --model_name erfnet_incremental_set12 --train_set 2 --num_epochs %num_epoch% --teachers erfnet_incremental_set1 %teacher_num_epoch% 1 --num_exp %num_exp% --num_shot %num_shot% --pseudo_label_mode --num_pseudo_labels %num_pseudo_labels% --dataset %dataset%
) else (
    python train_erfnet_incremental.py --model_name erfnet_incremental_set12 --train_set 2 --num_epochs %num_epoch% --teachers erfnet_incremental_set1 %teacher_num_epoch% 1 --num_exp %num_exp% --num_shot %num_shot% --dataset %dataset%
)

REM Stage 3, with second teacher
if "%enable_stage_3%"=="true" (
    if "%enable_pl%"=="true" (
        python train_erfnet_incremental.py --model_name erfnet_incremental_set123 --train_set 3 --num_epochs %num_epoch% --teachers erfnet_incremental_set12 %teacher_num_epoch% 2 --num_exp %num_exp% --num_shot %num_shot% --load_best_model --pseudo_label_mode --num_pseudo_labels %num_pseudo_labels% --dataset %dataset%
    ) else (
        python train_erfnet_incremental.py --model_name erfnet_incremental_set123 --train_set 3 --num_epochs %num_epoch% --teachers erfnet_incremental_set12 %teacher_num_epoch% 2 --num_exp %num_exp% --num_shot %num_shot% --load_best_model --dataset %dataset%
    )
)

endlocal
