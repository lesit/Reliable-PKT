import os
import time
import numpy as np
import pandas as pd

from make_model_input_data import MakeModelInputData
import  env_config

import sys
module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)

from src.misc import *

class TrainTestDataLoad:
    def __init__(self, 
                 is_train_phase,
                 logger):
        self.is_train_phase = is_train_phase
        self.logger = logger

        self.load_data_path = ""

        self.data_df = None

        self.data:dict = None

        self.input_index_len_dict = None

    def load_data(self,
                  n_problems,
                  max_seq_len,
                  max_code_len,
                  code_ast_path_dict,
                  code_path_df,
                  code_knowledge_emb,
                  dkt_features_dir):
        st = time.time()

        self.logger.info(f"TrainTestDataLoad.load_data: is_train_phase:{self.is_train_phase}, data:{dkt_features_dir}. start")

        make_input_data = MakeModelInputData(
            n_problems,
            max_seq_len,
            max_code_len,
            code_ast_path_dict,
            code_knowledge_emb,
            self.logger)

        self.input_index_len_dict = make_input_data.input_index_len_dict

        self.load_data_path = os.path.join(dkt_features_dir, "train_data.csv" if self.is_train_phase else "test_data.csv")

        self.data_df = pd.read_csv(self.load_data_path).astype({'student':str})
        self.data = make_input_data.get_data(self.data_df, code_path_df)

        self.logger.info(f"TrainTestDataLoad.load_data: data:{dkt_features_dir}. end elapse: {format_elapsed_time(time.time() - st)}\n")

    def get_train_fold_data(self, fold:int):
        if not self.is_train_phase:
            raise Exception("TrainTestDataLoad.get_train_fold_data: not train phase")
    
        df = self.data_df
        fold_train_df = df[df["fold"] != fold]
        fold_valid_df = df[df["fold"] == fold]

        train_student_ids = set(fold_train_df.student.to_list())
        valid_student_ids = set(fold_valid_df.student.to_list())

        train_list = [v for k,v in self.data.items() if k in train_student_ids]
        valid_list = [v for k,v in self.data.items() if k in valid_student_ids]
        
        train_shapes = set([v.shape for v in train_list])
        valid_shapes = set([v.shape for v in valid_list])
        if len(train_shapes) > 1 or len(valid_shapes) > 1:
            self.logger.info(f"invalid shapes: [{train_shapes}], [{valid_shapes}]")
            exit(-1)

        train_data = np.stack(train_list)
        valid_data = np.stack(valid_list)

        return train_data, valid_data
    
    def get_test_data(self):
        if self.is_train_phase:
            raise Exception("TrainTestDataLoad.get_test_data: train phase")

        test_data = np.stack(list(self.data.values()))

        return test_data

def make(is_train_phase,
         n_problems,
         max_seq_len,
         max_code_len,
         code_ast_path_dict,
         code_path_df,
         code_knowledge_emb,
         dkt_features_dir,
         logger) -> TrainTestDataLoad:
    
    data_load = TrainTestDataLoad(is_train_phase, logger)
    data_load.load_data(
        n_problems,
        max_seq_len,
        max_code_len,
        code_ast_path_dict,
        code_path_df,
        code_knowledge_emb,
        dkt_features_dir)

    return data_load
