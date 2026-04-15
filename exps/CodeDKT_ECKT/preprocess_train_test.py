import pandas as pd
import json
import numpy as np
import argparse
import os
import shutil
import torch

from tqdm import tqdm

import sys
import time

module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)

import  env_config

from exps import split_student_ids

import src.make_logger as make_logger

def attempt_preprocess(problem_ranks, problems_d, splited_student_ids, main_df, save_dir, logger):
    # main_df = main_df.sort_values(by=['ServerTimestamp'])   

    students = pd.unique(main_df["SubjectID"])

    logger.info(f"students.shape:{students.shape}")

    d = {}
    for s in students:
        d[s] = {}
        df = main_df[main_df["SubjectID"] == s]

        df = df.sort_values(by=['ServerTimestamp'])

        d[s]["length"] = len(df)    # 해당 학생의 실습 개수(execution 또는 submission)
        d[s]["problem_ranks"] = [str(problem_ranks[problem_id]) for problem_id in df["ProblemID"]]
        d[s]["Problems"] = [str(problems_d[problem_id]) for problem_id in df["ProblemID"]]

        results = (df["Score"]==1).astype(int)
        d[s]["Result"] = list(results.astype(str))
        d[s]["mean"] = results.mean()

        d[s]["CodeStates"] = [str(x) for x in df["CodeStateID"]]

    test_s = splited_student_ids["test"]
    train_fold_ids_list = splited_student_ids["train_folds"]

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    def save_feature_data(is_test, fold_ids_list, filename):
        def make_rows(fold, ids_list:np.array):
            ids_list = list(ids_list)
            values = []
            for s in ids_list:
                if s not in d:
                    logger.info(f"is_test:{is_test}: {s} not in data")
                    continue

                if d[s]['length']==0:
                    continue

                length = str(d[s]['length'])
                CodeStates = ",".join(d[s]['CodeStates'])
                problem_ranks = ",".join(d[s]['problem_ranks'])
                Problems = ",".join(d[s]['Problems'])
                Result = ",".join(d[s]['Result'])
                mean = d[s]["mean"]
                values.append([-1 if is_test else fold, s, length, CodeStates, problem_ranks, Problems, Result, mean])
            return values

        if is_test:
            values = make_rows(-1, fold_ids_list)
        else:
            values = []
            for fold, ids_list in enumerate(fold_ids_list):
                fold_rows = make_rows(fold, ids_list)
                values.extend(fold_rows)

        df = pd.DataFrame(data=values, columns=["fold", "student", "length", "CodeStates", "problem_ranks", "Problems", "Result", "Result_mean"])
        df.to_csv(os.path.join(save_dir, filename), index=False, encoding="utf-8-sig")

        mean = df.Result_mean.mean()
        logger.info(f"{filename}. result mean:{mean}")
        return df

    test_df = save_feature_data(True, test_s, "test_data.csv")
    train_df = save_feature_data(False, train_fold_ids_list, "train_data.csv")

    test_students = set(test_df["student"].unique())
    train_students = set(train_df["student"].unique())
    inter = test_students.intersection(train_students)
    assert len(inter) == 0

from src.misc import *

import make_cached_data
import code_ast_path

def make_train_test_cached_data(
        assignment,
        code_ast_path_dict,
        code_path_df,
        code_knowledge_emb,
        folds, 
        n_problems,
        max_seq_len,
        max_code_len,
        dkt_feature_dir,
        result_dir, 
        logger):

    logger.info(f"make_train_cached_data.start")
    st = time.time()

    mcd = make_cached_data.MakeCachedData(
        assignment,
        code_ast_path_dict,
        code_path_df,
        code_knowledge_emb,
        folds,
        n_problems,
        max_seq_len, 
        max_code_len,
        logger)

    logger.info(f"make_train_cached_data.end")

    mcd.make(dkt_feature_dir, result_dir)
    
    logger.info(f"make_train_cached_data.end elapse: {format_elapsed_time(time.time()-st)}")

import env_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../../dataset/CodeWorkout")
    parser.add_argument("--result_dir", type=str, default="../../results")
    args = parser.parse_args()

    print(args)
    dataset_root_dir = args.dataset_dir

    result_dir = args.result_dir
        
    result_dir = os.path.join(result_dir, "data")

    logger = make_logger.make("preprocess_train_test", save_dir=result_dir)
    logger.info(f"start")
    st = time.time()

    with open(os.path.join(result_dir, "problem_difficulty_map.json"), "r") as f:
        problem_difficulty_map = json.load(f)

    submission_path = os.path.join(dataset_root_dir, "MainTable.csv")
    submission_df = pd.read_csv(submission_path)
    submission_df = submission_df[submission_df["EventType"] == "Run.Program"]
    submission_df = submission_df.astype({"AssignmentID":'int32'})
    submission_df = submission_df.astype({"AssignmentID":str, "SubjectID":str, "ProblemID":str, "CodeStateID":str})
    print(f"submission_df: {submission_df.shape}")

    assignments_splited_student_ids, overall_student_ids = split_student_ids.load_ids(os.path.join(args.result_dir, "data"))

    n_folds = len(overall_student_ids["train_folds"])

    st_make_code_path = time.time()
    logger.info(f"make_code_path.start.")

    default_config = env_config.ModelConfig()

    max_seq_len = max(env_config.seq_len_space)
    max_code_len = default_config.max_code_len

    st_extract = time.time()

    code_knowledge_emb = torch.load(os.path.join(result_dir, 'code_knowledge_emb.tc'), weights_only=False)

    for assignment, splited_student_ids in assignments_splited_student_ids.items():
        code_path_df = pd.read_csv(os.path.join(result_dir, f"labeled_paths_{assignment}.tsv"),sep="\t")
        code_path_df = code_path_df.astype({"AssignmentID":str, "SubjectID":str, "CodeStateID":str})

        code_ast_path_dict:dict = code_ast_path.load(result_dir, assignment)

        assignment = str(assignment)

        assignment_dir = os.path.join(result_dir, "assignments", assignment)
        if not os.path.isdir(assignment_dir):
            os.makedirs(assignment_dir)
        dkt_feature_dir = os.path.join(assignment_dir, "DKTFeatures")
        if not os.path.isdir(dkt_feature_dir):
            os.makedirs(dkt_feature_dir)

        assignment_df = submission_df[submission_df['AssignmentID'] == assignment]

        problems = pd.unique(assignment_df["ProblemID"])
        problems_d = {k:v for (v,k) in enumerate(problems) }
        np.save(os.path.join(assignment_dir, "problems.npy"), problems_d)

        assignment_problem_ranks = problem_difficulty_map[str(assignment)]

        attempt_preprocess(assignment_problem_ranks,
                           problems_d, 
                           splited_student_ids, 
                           assignment_df, 
                           dkt_feature_dir, 
                           logger)

        make_train_test_cached_data(
            assignment,
            code_ast_path_dict,
            code_path_df,
            code_knowledge_emb,
            n_folds,
            len(problems_d),
            max_seq_len,
            max_code_len,
            dkt_feature_dir,
            assignment_dir,
            logger)

    logger.info(f"end.{time.time() - st}")
