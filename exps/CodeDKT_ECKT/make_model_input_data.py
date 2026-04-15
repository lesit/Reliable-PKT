import numpy as np
import pandas as pd
from tqdm import tqdm

import os

import sys
module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)

from code_ast_path import *

class MakeModelInputData:
    def __init__(self, 
                 n_problems, 
                 max_seq_len,
                 max_code_len,
                 code_ast_path_dict:dict,
                 code_knowledge_emb:dict,
                 logger):

        self.code_ast_path_dict = code_ast_path_dict
        self.code_knowledge_emb = code_knowledge_emb

        self.node_word_index = code_ast_path_dict['node_word_index']
        self.path_word_index = code_ast_path_dict['path_word_index']
        
        self.n_problems = n_problems
        self.max_seq_len = max_seq_len
        self.max_code_len = max_code_len

        difficulty_size = 1
        code_knowledge_emb_size = len(list(code_knowledge_emb.values())[0])

        score_idx_len = (0, 2*self.n_problems)

        code_idx_len = (score_idx_len[0] + score_idx_len[1], 3*self.max_code_len)

        difficulty_idx_len = (code_idx_len[0] + code_idx_len[1], difficulty_size)
        knowledge_idx_len = (difficulty_idx_len[0] + difficulty_idx_len[1], code_knowledge_emb_size)

        self.input_index_len_dict = {
            "score": score_idx_len,
            "code": code_idx_len,
            "difficulty": difficulty_idx_len,
            "knowledge": knowledge_idx_len
        }
        total_len = sum([x[1] for x in self.input_index_len_dict.values()])
        self.data_shape = [self.max_seq_len, total_len]

        self.score_indices = score_idx_len[0], score_idx_len[0]+score_idx_len[1]
        self.code_indices = code_idx_len[0], code_idx_len[0]+code_idx_len[1]
        self.difficulty_indices = difficulty_idx_len[0], difficulty_idx_len[0]+difficulty_idx_len[1]
        self.knowledge_indices = knowledge_idx_len[0], knowledge_idx_len[0]+knowledge_idx_len[1]

        self.logger = logger
        self.logger.info(f"MakeModelInputData: data_shape: {self.data_shape}")

    def get_data(self, data_df:pd.DataFrame, code_path_df:pd.DataFrame) -> dict:
        self.logger.info(f"MakeModelInputData.get_data... : data_df:{data_df.shape}. start")
        making_data_shape = [len(data_df), self.data_shape[0], self.data_shape[1]]
        self.logger.info(f"MakeModelInputData.get_data.end: making data shape: {making_data_shape}")

        student_data_dict = dict()
        
        empty_code_path_count = 0
        nan_code_path_count = 0

        for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
            # 한 학생에 대한 데이터
            problems = [int(q) for q in row.Problems.strip().split(',')]
            attempt_scores = [int(a) for a in row.Result.strip().split(',')]
            code_ids = [cs for cs in row.CodeStates.strip().split(',')]
            problem_ranks = [int(r) for r in row.problem_ranks.strip().split(',')]

            step_indices = list(range(len(attempt_scores)))

            if len(attempt_scores) > self.max_seq_len:
                step_indices = step_indices[-self.max_seq_len:]

            temp = np.zeros(shape=self.data_shape, dtype=np.float32) 

            extra = self.max_seq_len - len(step_indices)

            for step_idx, v_idx in enumerate(step_indices):
                c_idx = step_idx + extra
                if attempt_scores[v_idx] == 1:
                    temp[c_idx][problems[v_idx]] = 1
                else:
                    temp[c_idx][problems[v_idx] + self.n_problems] = 1

                code_id = code_ids[v_idx]
                difficulty = problem_ranks[v_idx]
                code_knowledge_emb = self.code_knowledge_emb[code_id]

                temp[c_idx][self.difficulty_indices[0]] = difficulty
                temp[c_idx][self.knowledge_indices[0]: self.knowledge_indices[1]] = code_knowledge_emb

                code = code_path_df[code_path_df['CodeStateID']==code_id]['RawASTPath']
                if code.empty:
                    self.logger.info(f"code empty: {row.student}. stemp_idx = {step_idx}, v_idx = {v_idx}")
                    empty_code_path_count += 1
                    continue    # not parsed because of syntax error, or no ast node because of too simple

                code_str = code.iloc[0]
                if type(code_str) == str and len(code_str)>0:
                    code_paths = code_str.split("@")
                    raw_features = convert_to_idx(self.node_word_index,
                                                  self.path_word_index,
                                                  code_paths, self.logger)
                    if len(raw_features) == 0:
                        self.logger.info(f"convert_to_idx not made raw_features: {row.student}. stemp_idx = {step_idx}, v_idx = {v_idx}")
                        empty_code_path_count += 1
                        continue

                    if len(raw_features) < self.max_code_len:
                        raw_features += [[0,0,0]]*(self.max_code_len - len(raw_features))    # padding
                    else:
                        raw_features = raw_features[:self.max_code_len]

                    features = np.array(raw_features, dtype=np.float32).reshape(-1, self.max_code_len*3)
                    temp[c_idx][self.code_indices[0] : self.code_indices[1]] = features
                else:
                    nan_code_path_count += 1     # not parsed because of syntax error, or no ast node because of too simple

            student_data_dict[row.student] = temp

            if len(student_data_dict) % 500 == 0:
                self.logger.info(f"MakeModelInputData.get_data: n: {len(student_data_dict)}")

        self.logger.info(f"empty_code_path_count:{empty_code_path_count}, nan_code_path_count:{nan_code_path_count}")

        data_shapes = set([v.shape for v in student_data_dict.values()])
        if len(data_shapes) != 1:
            self.logger.info(f"MakeModelInputData.get_data: invalid data_shapes:{data_shapes}")
            exit(-1)

        data_shapes = data_shapes.pop()

        made_data_shape = [len(student_data_dict), data_shapes[0], data_shapes[1]]
        self.logger.info(f"MakeModelInputData.get_data.end: made data shape {made_data_shape}")

        for idx in range(len(making_data_shape)):
            if made_data_shape[idx] != making_data_shape[idx]:
                self.logger.info(f"MakeModelInputData.get_data: invalid made_data_shape")
                exit(-1)

        return student_data_dict
