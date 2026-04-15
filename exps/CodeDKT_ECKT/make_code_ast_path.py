import os
import time
import shutil
import pandas as pd
import json
import numpy as np
import itertools

import sys

module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)

import src.make_logger as make_logger
from src.misc import *

from code_path_extract import path_extract
import code_ast_path

def make_code_path(code_path_length,
                   code_path_width,
                   main_df, 
                   save_dir, 
                   logger):
    logger.info("make_code_path. start")

    codepath_save_dir = os.path.join(save_dir, "making_code_path")
    if os.path.isdir(codepath_save_dir):
        shutil.rmtree(codepath_save_dir)
    os.makedirs(codepath_save_dir)            

    logger.info(f"main_df.shape:{main_df.shape}")

    st_extract = time.time()

    path_extract(main_df,
                    code_path_length=code_path_length, 
                    code_path_width=code_path_width,
                    save_dir=save_dir,
                    logger=logger,
                    logger_name=f"make_code_path", 
                    log_dir=codepath_save_dir)
    logger.info(f"make_code_path.end. elapse: {format_elapse_from(st_extract)}")


def save_code_ast_path(code_path_length, code_path_width, semester_splited_student_ids, save_dir, logger):
    logger.info(f"save_code_ast_path.start")
    for assignment_id, student_ids in semester_splited_student_ids.items():
        training_students = set()
        for student_ids in student_ids["train_folds"]:
            training_students.update(student_ids)

        code_path_df = pd.read_csv(os.path.join(save_dir, f"labeled_paths_{assignment_id}.tsv"),sep="\t")
        code_path_df = code_path_df.astype({"SubjectID":str})

        code_ast_path.save(code_path_length, code_path_width, code_path_df, training_students, save_dir, assignment_id, logger)

        logger.info(f"assignment:{assignment_id}, code_path_df:{code_path_df.shape} training_students:{len(training_students)}")

    logger.info(f"save_code_ast_path.end")

import sys
import time

from exps import split_student_ids

import src.make_logger as make_logger

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_len", type=int, default=8)
    parser.add_argument("--path_width", type=int, default=2)
    parser.add_argument("-k","--kfold", type=int, default=5)
    parser.add_argument("--dataset_dir", type=str, default="../../dataset/CodeWorkout")
    parser.add_argument("--result_dir", type=str, default="../../results")
    parser.add_argument("--skip_extract_exist", action="store_true")
    args = parser.parse_args()

    save_dir = os.path.join(args.result_dir, "data")
    # if args.skip_extract_exist:
    #     exist = code_ast_path.load(save_dir)
    #     if exist is not None:
    #         if exist.code_path_length == args.path_len and \
    #             exist.code_path_width == args.path_width:

    #             logger = make_logger.make("make_code_ast_path", save_dir=save_dir)
    #             logger.info(f"make_code_path. already exist {save_dir}")
    #             exit(-1)

    logger = make_logger.make("make_code_ast_path", save_dir=save_dir)
    logger.info(f"start")

    st_make_code_path = time.time()
    logger.info(f"start.")

    main_df = pd.read_csv(os.path.join(args.dataset_dir, "MainTable.csv"))
    main_df = main_df[main_df["EventType"] == "Run.Program"]
    main_df = main_df.astype({"AssignmentID":'int32'})

    code_df = pd.read_csv(os.path.join(args.dataset_dir, "LinkTables", "CodeStates.csv"))
    logger.info(f"code_df.shape:{code_df.shape}")

    main_df = main_df.merge(code_df, left_on="CodeStateID", right_on="CodeStateID")

    semester_splited_student_ids, overall_student_ids = split_student_ids.load_ids(os.path.join(args.result_dir, "data"))

    make_code_path(args.path_len,
                   args.path_width,
                   main_df, 
                   save_dir, 
                   logger
                   )
    
    save_code_ast_path(args.path_len,
                       args.path_width,
                       semester_splited_student_ids, 
                       save_dir,
                       logger)
    logger.info(f"end. elapse: {format_elapsed_time(time.time() - st_make_code_path)}")
