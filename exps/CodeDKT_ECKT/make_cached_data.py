
import os
import make_train_test_data
import time
import numpy as np
import gc
import pandas as pd
import json
import re

import sys
module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)

from src.misc import *
import code_ast_path

def get_input_info_dict_path(result_dir):
    return os.path.join(result_dir, f"input_info.json")

def get_cached_data_info(is_train_phase, result_dir):
    traintest_data_dir = os.path.join(result_dir, f"traintest_data")
    if not os.path.exists(traintest_data_dir):
        return None

    input_info_dict_path = get_input_info_dict_path(result_dir)
    if not os.path.isfile(input_info_dict_path):
        return None

    n_folds = None

    if is_train_phase:
        train_fold_list = []
        valid_fold_list = []
        for fold_data_name in sorted(os.listdir(traintest_data_dir)):
            fold_data_path = os.path.join(traintest_data_dir, fold_data_name)
            if not os.path.isfile(fold_data_path):
                continue

            train_match = re.fullmatch(r"train_data_fold(\d+)\.npy", fold_data_name)
            if train_match:
                fold = int(train_match.group(1))
                train_fold_list.append(fold)

            valid_match = re.fullmatch(r"valid_data_fold(\d+)\.npy", fold_data_name)
            if valid_match:
                fold = int(valid_match.group(1))
                valid_fold_list.append(fold)

        if len(train_fold_list) == 0 or len(train_fold_list) != len(valid_fold_list):
            return None

        train_fold_list = sorted(train_fold_list)
        valid_fold_list = sorted(train_fold_list)
            
        last_fold = train_fold_list[-1]
        if valid_fold_list[-1] != last_fold:
            return None
        
        n_folds = last_fold + 1
        if len(train_fold_list) != n_folds:
            return None

    else:
        cache_test_data_path = os.path.join(os.path.abspath(traintest_data_dir), f"test_data.npy")
        if not os.path.isfile(cache_test_data_path):
            return None

    with open(input_info_dict_path, "r") as fj:
        input_info_dict = json.load(fj)
    return traintest_data_dir, input_info_dict, n_folds

class MakeCachedData:
    def __init__(self,
                 assignment,
                 code_ast_path_dict:dict,
                 code_path_df:pd.DataFrame,
                 code_knowledge_emb,
                 folds,
                 n_problems,
                 max_seq_len, 
                 max_code_len,
                 logger):
        self.assignment = assignment

        self.code_ast_path_dict:dict = code_ast_path_dict
        self.code_path_df:pd.DataFrame = code_path_df

        self.code_knowledge_emb = code_knowledge_emb

        self.n_problems = n_problems
        self.max_seq_len = max_seq_len
        self.max_code_len = max_code_len

        self.folds = folds

        self.logger = logger

    def make(self, dkt_feature_dir, save_dir) -> dict:
        cached_data_info = get_cached_data_info(True, save_dir)
        if cached_data_info is not None:
            train_data_dir, input_index_len_dict, n_folds = cached_data_info
            if n_folds == self.folds:
                cached_data_info = get_cached_data_info(False, save_dir)
                if cached_data_info is not None:
                    self.logger.info(f"has already done")
                    return cached_data_info

        import shutil
        traintest_data_dir = os.path.join(save_dir, f"traintest_data")
        if os.path.exists(traintest_data_dir):
            shutil.rmtree(traintest_data_dir)

        os.makedirs(traintest_data_dir)

        input_index_len_dict = self.__make(True, dkt_feature_dir, traintest_data_dir)
        self.__make(False, dkt_feature_dir, traintest_data_dir)

        input_info_dict_path = get_input_info_dict_path(save_dir)
        input_info_dict = {
            "code_ast_path_info":{
                "node_vocab_size": self.code_ast_path_dict["node_vocab_size"],
                "path_vocab_size": self.code_ast_path_dict["path_vocab_size"]
            },
            "input_index_len": input_index_len_dict  
        }
        with open(input_info_dict_path, "w") as fj:
            json.dump(input_info_dict, fj, indent=3)

    def __make(self, is_train_phase, dkt_feature_dir, save_dir) -> dict:
        self.logger.info(f"MakeCachedData.make.is_train_phase:{is_train_phase}, {save_dir} from {dkt_feature_dir}.start")

        st = time.time()

        gc.collect()

        data_load:make_train_test_data.TrainTestDataLoad = make_train_test_data.make(
            is_train_phase,
            self.n_problems,
            self.max_seq_len,
            self.max_code_len,
            self.code_ast_path_dict,
            self.code_path_df,
            self.code_knowledge_emb,
            dkt_feature_dir,
            self.logger)

        input_index_len_dict = data_load.input_index_len_dict

        if is_train_phase:
            for fold in range(self.folds):
                self.logger.info(f"MakeCachedData.make: data_load.get_fold_data {fold}.start")

                fold_train_data, fold_valid_data = data_load.get_train_fold_data(fold)
                fold_train_shape = fold_train_data.shape
                fold_valid_shape = fold_valid_data.shape
                self.logger.info(f"MakeCachedData.make: saving data shape: {fold_train_shape}, {fold_valid_shape}")

                cache_train_data_path = os.path.join(os.path.abspath(save_dir), f"train_data_fold{fold}.npy")
                np.save(cache_train_data_path, fold_train_data)

                cache_valid_data_path = os.path.join(os.path.abspath(save_dir), f"valid_data_fold{fold}.npy")
                np.save(cache_valid_data_path, fold_valid_data)

                del fold_train_data
                del fold_valid_data

                fold_train_data = np.load(cache_train_data_path, allow_pickle=True)
                fold_valid_data = np.load(cache_valid_data_path, allow_pickle=True)

                assert fold_train_shape == fold_train_data.shape
                assert fold_valid_shape == fold_valid_data.shape
                self.logger.info(f"MakeCachedData.make: loaded data shape: {fold_train_data.shape}, {fold_valid_data.shape}")

                del fold_train_data
                del fold_valid_data
                gc.collect()

                self.logger.info(f"MakeCachedData.make: data_load.get_fold_data {fold}.end")
        else:
            test_data = data_load.get_test_data()
            valid_shape = test_data.shape
            self.logger.info(f"MakeCachedData.make: saving data shape: {valid_shape}")

            cache_data_path = os.path.join(os.path.abspath(save_dir), f"test_data.npy")
            np.save(cache_data_path, test_data)

            del test_data

            test_data = np.load(cache_data_path, allow_pickle=True)
            assert valid_shape == test_data.shape
            self.logger.info(f"MakeCachedData.make: loaded data shape: {test_data.shape}")

            del test_data

            gc.collect()

        del data_load
        gc.collect()

        self.logger.info(f"MakeCachedData.make.{save_dir} from {dkt_feature_dir}.end: elapse:{format_elapsed_time(time.time()-st)}")

        return input_index_len_dict
