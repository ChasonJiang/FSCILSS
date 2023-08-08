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

:: Stage 1, first Teacher
if "%enable_stage_1%"=="true" (
    python evaluate_erfnet.py --load_model_name erfnet_incremental_set1 --train_set 1 --weights_epoch %teacher_num_epoch% --task_to_val 1 --eval_base_model --save_evaluation_results
)

:: Stage 2, with first teacher
if "%enable_pl%"=="true" (
    python evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch %teacher_num_epoch% --task_to_val 1 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --pseudo_label_mode --num_pseudo_labels %num_pseudo_labels% --num_shot %num_shot%
    python evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch %teacher_num_epoch% --task_to_val 2 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --pseudo_label_mode --num_pseudo_labels %num_pseudo_labels% --num_shot %num_shot%
    python evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch %teacher_num_epoch% --task_to_val 12 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --pseudo_label_mode --num_pseudo_labels %num_pseudo_labels% --num_shot %num_shot%
) else (
    python evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch %teacher_num_epoch% --task_to_val 1 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --num_shot %num_shot%
    python evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch %teacher_num_epoch% --task_to_val 2 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --num_shot %num_shot%
    python evaluate_erfnet.py --load_model_name erfnet_incremental_set12 --train_set 12 --weights_epoch %teacher_num_epoch% --task_to_val 12 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --num_shot %num_shot%
)

:: Stage 3, with second teacher
if "%enable_stage_3%"=="true" (
    if "%enable_pl%"=="true" (
        python evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch %teacher_num_epoch% --task_to_val 1 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --pseudo_label_mode --num_pseudo_labels %num_pseudo_labels% --num_shot %num_shot%
        python evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch %teacher_num_epoch% --task_to_val 2 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --pseudo_label_mode --num_pseudo_labels %num_pseudo_labels% --num_shot %num_shot%
        python evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch %teacher_num_epoch% --task_to_val 3 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --pseudo_label_mode --num_pseudo_labels %num_pseudo_labels% --num_shot %num_shot%
        python evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch %teacher_num_epoch% --task_to_val 12 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --pseudo_label_mode --num_pseudo_labels %num_pseudo_labels% --num_shot %num_shot%
        python evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch %teacher_num_epoch% --task_to_val 123 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --pseudo_label_mode --num_pseudo_labels %num_pseudo_labels% --num_shot %num_shot%
    ) else (
        python evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch %teacher_num_epoch% --task_to_val 1 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --num_shot %num_shot%
        python evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch %teacher_num_epoch% --task_to_val 2 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --num_shot %num_shot%
        python evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch %teacher_num_epoch% --task_to_val 3 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --num_shot %num_shot%
        python evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch %teacher_num_epoch% --task_to_val 12 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --num_shot %num_shot%
        python evaluate_erfnet.py --load_model_name erfnet_incremental_set123 --train_set 123 --weights_epoch %teacher_num_epoch% --task_to_val 123 --multi_exp %num_exp% --save_evaluation_results --dataset %dataset% --num_shot %num_shot%
    )
)
