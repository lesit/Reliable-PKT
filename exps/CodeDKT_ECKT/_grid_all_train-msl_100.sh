#!/bin/bash

BASE_CMD="python -u train_hp_tune_cv.py --all_grid --msl 100"

if [ -n "$1" ]; then
    BASE_CMD="$BASE_CMD --compare_dir $1"
fi

nohup sh -c "
    $BASE_CMD --model_type dkt && \
    $BASE_CMD --model_type codedkt --att_dim_valid && \
    $BASE_CMD --model_type codedkt --att_dim_valid --has_w0 && \
    $BASE_CMD --model_type eckt --att_dim_valid && \
    $BASE_CMD --model_type eckt --att_dim_valid --has_w0
" > sh_train_msl${MAX_SEQ_LEN}.log 2>&1 &
