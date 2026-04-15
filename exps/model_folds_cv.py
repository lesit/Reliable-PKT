import os
import argparse
import json
import copy
import pandas as pd
import numpy as np
import time
import torch

import sys
module_path = os.path.abspath("../")
if module_path not in sys.path:
	sys.path.append(module_path)

import src.make_logger as make_logger
from src.misc import *

import pandas as pd

import json
def train_cv_on_data(
        trainable_func,
        eval_fold_func,
        eval_batch_size,
        seed,
        num_epochs,
        es_delta,
        es_patience,
        hp_config, 
        train_valid_data_dir,
        n_folds, 
        result_dir,
        logger,
        **kwargs):
    logger.info(f"train_cv_on_data.start result_dir:{result_dir}")

    eval_result_path = os.path.join(result_dir, f"eval_result.csv")
    if os.path.isfile(eval_result_path):
        eval_df = pd.read_csv(eval_result_path)
        return eval_df

    st = time.time()
    
    fold_model_path_list = []

    n_gpus = torch.cuda.device_count()

    import multiprocessing
    manager = multiprocessing.Manager()
    shared_result_path_dict = manager.dict()

    process_list = []

    from collections import defaultdict
    gpu_fold_list = defaultdict(list)

    gpu_id = 0
    for fold in range(n_folds):
        gpu_fold_list[gpu_id].append(fold)
        gpu_id = (gpu_id + 1) % n_gpus

    for gpu_id in range(n_gpus):
        process = multiprocessing.Process(target=train_folds_cv,
                                          args=[
                                              trainable_func,
                                              eval_fold_func,
                                              gpu_id,
                                              eval_batch_size,
                                              gpu_fold_list[gpu_id],
                                              seed,
                                              num_epochs,
                                              es_delta,
                                              es_patience,
                                              hp_config,
                                              train_valid_data_dir,
                                              result_dir,
                                              shared_result_path_dict,
                                          ],
                                          kwargs=kwargs
                                          )
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    fold_eval_df_list = []
    for fold in range(n_folds):
        save_model_path, eval_df_path = shared_result_path_dict[fold]
        fold_model_path_list.append(save_model_path)

        fold_eval_df = pd.read_csv(eval_df_path)
        fold_eval_df_list.append(fold_eval_df)

    total_df = pd.concat(fold_eval_df_list, ignore_index=True)

    performance_mean = np.mean(total_df.values,axis=0)
    performance_mean = [f"{float(x):.4f}" for x in performance_mean]
    performance_std = np.std(total_df.values,axis=0)
    performance_std = [f"{float(x):.4f}" for x in performance_std]
    performance_mean_std = [f"{mean},{std}" for mean,std in zip(performance_mean, performance_std)]

    performances = []
    performances.append(["mean_std"] + performance_mean_std)
    for fold, performance in enumerate(total_df.values):
        performances.append([str(fold)] + list(performance))
                
    eval_df = pd.DataFrame(performances, columns=['fold'] + list(total_df.columns))

    data_unit_name = kwargs.get("data_unit_name")
    data_unit_type = kwargs.get("data_unit_type")
    if data_unit_name is not None and data_unit_type is not None:
        eval_df.insert(0, data_unit_name, data_unit_type)
    eval_df.to_csv(eval_result_path , index=False)

    logger.info(f"train_cv_on_data. end. result_dir:{result_dir},  elapse:{format_elapse_from(st)}\n")
    return eval_df

def train_folds_cv(
        trainable_func,
        eval_fold_func,
        gpu_id,
        eval_batch_size,
        folds,
        seed,
        num_epochs,
        es_delta,
        es_patience,
        hp_config,
        train_valid_data_dir,
        train_save_root_dir,
        shared_result_path_dict,
        logger=None,
        **kwargs):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if logger is None:
        mp_logger = make_logger.make(f"train_folds_cv_mp_gpu_id{gpu_id}", time_filename=False, save_dir=train_save_root_dir)
    else:
        mp_logger = logger

    mp_logger.info(f"train_folds_cv.folds:{folds}.start")

    for fold in folds:
        mp_logger.info(f"fold:{fold}. start")

        train_fold_save_dir = os.path.join(train_save_root_dir, f"fold_{fold}")
        save_model_path = os.path.join(train_fold_save_dir, "best_model.ckpt")
        if not os.path.isfile(save_model_path):
            mp_logger.info(f"trainable_func:{fold}. save dir:{train_fold_save_dir} start")
            try:
                trainable_func(
                    hp_config,
                    is_hp_tuning = False,
                    seed = seed,
                    fold = fold,
                    num_epochs = num_epochs,
                    save_dir = train_fold_save_dir,
                    save_model_path = save_model_path,
                    es_delta = es_delta,
                    es_patience = es_patience,
                    train_data_dir = train_valid_data_dir,
                    **kwargs)
            except Exception as e:
                make_logger.write_exception_log(mp_logger, e, f"main.train_fold. trainable_func:{fold}")
                exit(-1)
            mp_logger.info(f"trainable_func:{fold}. save dir:{train_fold_save_dir} end")

        eval_df_path = os.path.join(train_fold_save_dir, "eval_df.csv")
        if not os.path.isfile(eval_df_path):
            mp_logger.info(f"eval_fold_func:{fold}. save_model_path:{save_model_path} start")
            try:
                dres = eval_fold_func(
                    eval_batch_size,
                    train_valid_data_dir, 
                    save_model_path, 
                    mp_logger, 
                    log_prefix=f"{fold}",
                    **kwargs)
            except Exception as e:
                make_logger.write_exception_log(mp_logger, e, f"main.train_fold. eval_fold_func:{fold}")
                exit(-1)
            mp_logger.info(f"eval_fold_func:{fold}. save_model_path:{save_model_path}. dres:{dres}. end")

            if dres is None:
                mp_logger.info(f"no eval result")
                exit(-1)
            
            values = [list(dres.values())]
            eval_df = pd.DataFrame(data=values, columns=list(dres.keys()))
            eval_df.to_csv(eval_df_path, index=False)

        shared_result_path_dict[fold] = save_model_path, eval_df_path

        mp_logger.info(f"fold:{fold}. save dir:{train_fold_save_dir} end.\n")
    mp_logger.info(f"train_folds_cv.folds:{folds}.end\n")

