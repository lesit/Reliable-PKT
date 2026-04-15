#!/bin/bash

nohup sh -c "
    python -u run.py --gpu_id 0 --dkt && \
    python -u run.py --gpu_id 0
" > run_gpu_0.out 2>&1 &

nohup sh -c "
    python -u run.py --gpu_id 1 --dkt --sort_by_time && \
    python -u run.py --gpu_id 1 --sort_by_time && \
    python -u run.py --gpu_id 1 --sort_by_time --valid_att_dim
" > run_gpu_1.out 2>&1 &
