import os
import pandas as pd
import numpy as np
import pickle
import copy
from collections import defaultdict

import sys
module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)

import src.make_logger as make_logger

def create_word_index_table(vocab):
    """
    Creating word to index table
    Input:
    vocab: list. The list of the node vocabulary

    """
    ixtoword = {}
    # period at the end of the sentence. make first dimension be end token
    ixtoword[0] = 'END'
    ixtoword[1] = 'UNK'
    wordtoix = {}
    wordtoix['END'] = 0
    wordtoix['UNK'] = 1
    ix = 2
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    return wordtoix, ixtoword

from code_path_extract import *

class CodeAstPath():
    def __init__(self, code_path_length, code_path_width, code_path_df:pd.DataFrame):
        self.code_path_length = code_path_length
        self.code_path_width = code_path_width

        self.code_path_df:pd.DataFrame = code_path_df
        self.node_hist:dict = None
        self.path_hist:dict = None
        self.node_word_index:dict = None
        self.path_word_index:dict = None

    def create_node_path_index(self, training_students, logger):
        # 학습에 사용된 코드에 대해서만 word index를 구해서, test 시에는 학습에 대해서 평가할 수 있도록 한다
        all_training_code = self.code_path_df[self.code_path_df['SubjectID'].isin(training_students)]['RawASTPath']

        logger.info("CodeAstPath.create_node_path_index. split codes")
        separated_code = []
        n_empty_code = 0
        for code in all_training_code:
            if type(code) == str:
                separated_code.append(code.split("@"))
            else:
                n_empty_code += 1

        logger.info(f"CodeAstPath.create_node_path_index. split codes. empty:{n_empty_code}")

        logger.info("CodeAstPath.make code and node path")
        node_hist = {}
        path_hist = {}
        for paths in separated_code:
            starting_nodes = [p.split(",")[0] for p in paths]
            path = [p.split(",")[1] for p in paths]
            ending_nodes = [p.split(",")[2] for p in paths]
            nodes = starting_nodes + ending_nodes
            for n in nodes:
                if not n in node_hist:
                    node_hist[n] = 1
                else:
                    node_hist[n] += 1
            for p in path:
                if not p in path_hist:
                    path_hist[p] = 1
                else:
                    path_hist[p] += 1
                    
        # small frequency then abandon, for node and path
        valid_node = [node for node, count in node_hist.items()]
        valid_path = [path for path, count in path_hist.items()]

        # create ixtoword and wordtoix lists
        node_word_index, node_index_word = create_word_index_table(valid_node)
        path_word_index, path_index_word = create_word_index_table(valid_path)

        self.node_hist = node_hist
        self.path_hist = path_hist
        self.node_word_index = node_word_index
        self.node_index_word = node_index_word
        self.path_word_index = path_word_index
        self.path_index_word = path_index_word

def get_code_ast_path_info_path(save_dir, assignment):
    return os.path.join(save_dir, f"code_ast_path_{assignment}.json")

def save(code_path_length, code_path_width, code_path_df, training_students, save_dir, assignment, logger):
    code_ast_path_inst = CodeAstPath(code_path_length, code_path_width, code_path_df)
    try:
        code_ast_path_inst.create_node_path_index(training_students, logger)
    except Exception as e:
        make_logger.write_exception_log(logger, e, "code_ast_path.save")

    code_ast_path_dict = {
        "code_path_length": code_path_length,
        "code_path_width": code_path_width,

        "node_vocab_size": len(code_ast_path_inst.node_word_index),
        "path_vocab_size": len(code_ast_path_inst.path_word_index),

        "node_word_index": code_ast_path_inst.node_word_index,
        "path_word_index": code_ast_path_inst.path_word_index,

        "node_hist": code_ast_path_inst.node_hist,
        "path_hist": code_ast_path_inst.path_hist
    }

    save_path = get_code_ast_path_info_path(save_dir, assignment)
    with open(save_path, 'w') as f:
        json.dump(code_ast_path_dict, f, indent=3)

        logger.info(f"save_code_node_path. save to {save_path}")

def load(data_dir, assignment) -> CodeAstPath:
    saved_path = get_code_ast_path_info_path(data_dir, assignment)
    if not os.path.isfile(saved_path):
        return None
    
    with open(saved_path, 'rb') as f:
        code_ast_path_dict = json.load(f)
    
    return code_ast_path_dict

def convert_to_idx(node_word_index,
                   path_word_index,
                   sample, logger):
    """
    Converting to the index 
    Input:
    sample: list. One single training sample, which is a code, represented as a list of neighborhoods.
    node_word_index: dict. The node to word index dictionary.
    path_word_index: dict. The path to word index dictionary.

    """
    unk_node_word_idx = node_word_index['UNK']
    unk_path_word_idx = path_word_index['UNK']

    skipped_low_freqs = []
    sample_index = []
    for line in sample:
        components = line.split(",")
        if len(components)<3:
            a = 0
        path_word = components[1]

        path_node = path_word_index.get(path_word)
        if path_node is None:
            path_node = unk_path_word_idx

        starting_node = node_word_index.get(components[0], unk_node_word_idx)
        ending_node = node_word_index.get(components[2], unk_node_word_idx)

        sample_index.append([starting_node, path_node, ending_node])

    if len(sample_index) == 0:
        logger.info(f"convert_to_idx: sample:{len(sample)}")
        if len(skipped_low_freqs)>0:
            logger.info(f"convert_to_idx.skipped_low_freqs: {len(skipped_low_freqs)}")

    # assert len(sample_index)>0
    return sample_index
