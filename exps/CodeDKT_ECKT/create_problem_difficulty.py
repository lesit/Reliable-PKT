import pandas as pd
import json
import torch
import os
import argparse

import sys

module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)

def save_problem_difficulty(train_student_ids:set, csv_path, save_path):
    df = pd.read_csv(csv_path)
    df = df[df["EventType"] == "Run.Program"]
    df = df.astype({"AssignmentID":'int32'})
    df = df[df['SubjectID'].isin(train_student_ids)]

    df['result'] = (df['Score'] == 1).astype(int)

    assignment_list = sorted(list(df["AssignmentID"].unique()))

    assignment_difficulty = {}
    for assignment in assignment_list:
        assignment_df = df[df["AssignmentID"]==assignment]

        prob_stats = assignment_df.groupby('ProblemID')['result'].mean().reset_index()
        prob_stats = prob_stats.sort_values(by='result', ascending=True).reset_index(drop=True)
        
        prob_stats['difficulty_rank'] = range(len(prob_stats))
        
        difficulty_map = dict(zip(prob_stats['ProblemID'], prob_stats['difficulty_rank']))
        assignment_difficulty[int(assignment)] = difficulty_map

    with open(save_path, "w") as fout:
        json.dump(assignment_difficulty, fout, ensure_ascii=False, indent=3)

    print(f"completed: {assignment_difficulty}")
    return difficulty_map

from exps import split_student_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../../dataset/CodeWorkout")
    parser.add_argument("--result_dir", type=str, default="../../results")
    args = parser.parse_args()

    semester_splited_student_ids, overall_student_ids = split_student_ids.load_ids(os.path.join(args.result_dir, "data"))
    train_fold_ids_list = overall_student_ids["train_folds"]
    train_student_ids = set()
    for fold_ids in train_fold_ids_list:
        train_student_ids.update(fold_ids)

    submit_path = os.path.join(args.dataset_dir, "MainTable.csv")
    save_path = os.path.join(args.result_dir, "data", "problem_difficulty_map.json")
    save_problem_difficulty(train_student_ids, submit_path, save_path)

    # backup = torch.load(os.path.join(args.save_dir, "problem_difficulty_map_backup.pt"))
    # current = torch.load(save_path)
    
    # diff = []
    # for problem, rank in current.items():
    #     backup_rank = backup[problem]
    #     if backup_rank != rank:
    #         diff.append((problem, backup_rank, rank))
    # print(f"is equal: {backup == current}")
