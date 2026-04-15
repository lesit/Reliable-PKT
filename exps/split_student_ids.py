
import os
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold

import sys
module_path = os.path.abspath("../")
if module_path not in sys.path:
	sys.path.append(module_path)

import src.make_logger as make_logger

fname = "student_folds_split.csv"

def split_ids(dataset_root_dir, save_data_root_dir, kfold, logger):
    if not os.path.isdir(save_data_root_dir):
        os.makedirs(save_data_root_dir)            

    main_df = pd.read_csv(os.path.join(dataset_root_dir, "MainTable.csv"))
    main_df = main_df[main_df["EventType"] == "Run.Program"]
    main_df = main_df.astype({"AssignmentID":'int32'})

    logger.info(f"main_df: {main_df.shape}")

    # all_test_s = set()
    # all_fold_s = [set() for x in range(kfold)]

    assignment_list = sorted(list(main_df["AssignmentID"].unique()))

    fold_students = defaultdict(list)
    for assignment in assignment_list:
        logger.info(f"assignment: {assignment}")
        semester_submissions_df = main_df[main_df["AssignmentID"] == assignment]

        students = semester_submissions_df["SubjectID"].unique()
        logger.info(f"total students. num: {len(students)}")

        train_val_s, test_s = train_test_split(students, test_size=0.2, random_state=1)

        logger.info(f"test students. num: {len(test_s)}")

        # codeworkout 데이터는 학생들이 여러 assignment에 속해 있다.
        # dup = all_test_s.intersection(test_s)
        # assert len(dup) == 0
        # all_test_s.update(test_s)

        for student_id in test_s:
            fold_students[-1].append([assignment, student_id, f"{assignment}_{student_id}"])

        kf = KFold(n_splits=kfold, shuffle=True, random_state=1024)

        logger.info(f"train students. num: {len(train_val_s)}")
        for fold, (_, val_index) in enumerate(kf.split(train_val_s)):
            fold_s = train_val_s[val_index]

            for student_id in fold_s:
                fold_students[fold].append([assignment, student_id, f"{assignment}_{student_id}"])

            logger.info(f"{fold} fold. num: {len(fold_s)}")

            # assert len(all_fold_s[fold].intersection(fold_s)) == 0
            # all_fold_s[fold].update(fold_s)
        logger.info("")

    fold_student_rows = []
    for fold, students in fold_students.items():
        for student in students:
            fold_student_rows.append([fold] + student)

    df = pd.DataFrame(fold_student_rows, columns=["fold", "assignment", "student_id", "assignment_student"])
    df.to_csv(os.path.join(save_data_root_dir, fname), index=False)

def load_ids(saved_dir="../results/data"):
    full_df = pd.read_csv(os.path.join(saved_dir, fname))
    full_df = full_df.astype({"assignment":str, "student_id":str})

    fold_list = full_df[full_df["fold"]>=0]["fold"].unique()

    def make_fold_student_ids(df):
        test_s = df[df["fold"] == -1]["student_id"].to_list()

        train_fold_ids_list = []
        for fold in sorted(fold_list):
            fold_s = df[df["fold"] == fold]["student_id"].to_list()
            train_fold_ids_list.append(fold_s)

        return {
            "test": test_s,
            "train_folds": train_fold_ids_list
        }

    assignment_student_ids = dict()
    assignment_list = full_df["assignment"].unique()
    for assignment in sorted(assignment_list):
        assignment_df = full_df[full_df["assignment"] == assignment]

        assignment_student_ids[int(assignment)] = make_fold_student_ids(assignment_df)

    overall_student_ids = make_fold_student_ids(full_df)
    return assignment_student_ids, overall_student_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../dataset/CodeWorkout")
    parser.add_argument("-k","--kfold", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="../results/data")
    args = parser.parse_args()

    logger = make_logger.make("split_student_ids", save_dir=args.save_dir)
    logger.info(f"start")

    split_ids(args.dataset_dir, args.save_dir, args.kfold, logger)

    logger.info(f"end")
